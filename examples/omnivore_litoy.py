import time
from pathlib import Path, PosixPath
import json
from typing import List, Union, Optional, Dict
from typeguard import typechecked
import fire
from dateutil.relativedelta import relativedelta
from datetime import datetime
import logging
import asyncio
import concurrent.futures
import random
import os

from tqdm import tqdm
from omnivoreql import OmnivoreQL
from mini_LiTOY import mini_LiTOY

# Configure logging
logging.basicConfig(filename='omnivore_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()


MAX_REQUEST_TRIALS = 5
MAX_CONCURRENCY = 10

@typechecked
def _load_api_key() -> str:
    if "OMNIVORE_API_KEY" not in os.environ:
        raise Exception("No OMNIVORE_API_KEY found in environment, and not given as arugment")
    elif not os.environ["OMNIVORE_API_KEY"]:
        raise Exception("Empty OMNIVORE_API_KEY found in environment")
    omnivore_api_key = os.environ["OMNIVORE_API_KEY"]
    return omnivore_api_key

default_dict = mini_LiTOY.LockedDict({
    "entry": None,
    "id": None,
    "metadata": {},

    "g_n_comparison": 0,
    "g_ELO": mini_LiTOY.ELO_default,
    "all_ELO": {
        q: mini_LiTOY.LockedDict({"q_ELO": mini_LiTOY.ELO_default, "q_n_comparison": 0})
        for q in mini_LiTOY.questions
    },
})
metadata_keys = [
    "description",
    "siteName",
    "labels",
    "originalArticleUrl",
    "url",
    "wordsCount",
    "savedAt",
    "readingProgressAnchorIndex",
    "readingProgressPercent",
    "readingProgressTopPercent",
]

@typechecked
def exec_query(base_query: str, d1: str, d2: str, omnivore_api_key: str, pbar: tqdm) -> Dict:
    "synchronous data fetcher"
    client = OmnivoreQL(omnivore_api_key)
    query = base_query.replace("$start_date", d1).replace("$end_date", d2)
    trial = 0
    time.sleep(random.random() * 10 / 2)   # randomize 0-5s the request start
    while trial < MAX_REQUEST_TRIALS:
        trial += 1
        try:
            d = client.get_articles(
                    limit=1000,
                    query=query,
            )
            break
        except Exception as err:
            tqdm.write(
                f"Error when loading articles for query '{query}'\nError: '{err}'"
            )
            time.sleep(5)

    edges = d["search"]["edges"]
    tqdm.write(f" * {d1}->{d2}: found {len(edges)} articles among {len(edges)}")
    pbar.update(1)
    return d

@typechecked
def update_js(
    json_file_to_update: Union[str, PosixPath],
    omnivore_api_key: Optional[str] = None,
    start_date: Union[str, datetime] = "2023-04-01",
    base_query: str = "in:inbox -type:highlights sort:saved saved:$start_date..$end_date",
    time_window: int = 7,
    ):
    if omnivore_api_key is None:
        omnivore_api_key = _load_api_key()

    log.info("Starting omnivore update")
    try:
        client = OmnivoreQL(omnivore_api_key)
        labels = client.get_labels()["labels"]["labels"]
    except Exception as err:
        raise Exception(f"Error when logging to OmnivoreQL then loading labels: '{err}'")

    # generates all date ranges from start_date to today
    tmr = datetime.today() + relativedelta(days=1)
    end_date = f"{tmr.year}-{tmr.month}-{tmr.day}"
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    current_date = start_date
    dates = []
    while current_date <= end_date:
        newdate = [
                (current_date - relativedelta(days=1)).strftime("%Y-%m-%d"),
                (current_date + relativedelta(days=time_window)).strftime("%Y-%m-%d"),
        ]
        current_date += relativedelta(days=time_window)
        dates.append(newdate)
    dates = dates[::-1]

    json_file_to_update =  Path(json_file_to_update)
    if json_file_to_update.exists():
        try:
            json_articles = json.load(Path(json_file_to_update).open("r"))
        except Exception as err:
            raise Exception(f"Error when loading {json_file_to_update}: '{err}'")
    else:
        json_articles = []
    assert isinstance(json_articles, list), "loaded json is not a list"
    assert all(isinstance(article, dict) for article in json_articles), "loaded json is not a list of dict"

    # execute async queries
    pbar = tqdm(total=len(dates), desc="Querying", unit="date_ranges")
    async def main():
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    exec_query,
                    base_query,
                    d1,
                    d2,
                    omnivore_api_key,
                    pbar,
                )
                for d1, d2 in dates
            ]
            results = await asyncio.gather(*tasks)
        return results
    results = asyncio.run(main())

    assert len(results) == len(dates), f"Number of results={len(results)} but number of date ranges: {len(dates)}"

    present_ids = [article["id"] for article in json_articles]

    for idate, _dat in tqdm(enumerate(dates), total=len(dates)):
        d1, d2 = _dat
        d = results[idate]
        edges = d["search"]["edges"]

        if len(edges) >= 100:
            raise Exception(f"Found {len(edges)} articles for date '{d1}..{d2}', this is above 100 so use a lower time_window.")

        n_new = 0
        for e in edges:
            n = e["node"]
            if n["id"] in present_ids:
                continue
            present_ids.append(n["id"])

            new = default_dict.copy()
            new["id"] = n["id"]
            new["entry"] = n['title']
            if n["author"] and n["author"] not in new["entry"]:
                new["entry"] += f"\nby {n['author']}"
            if n["siteName"] and n["siteName"] not in new["entry"]:
                new["entry"] += f"\non {n['siteName']}"
            new["metadata"] = {k: n[k] for k in metadata_keys}
            # don't save labels as dict
            new["metadata"]["labels"] = ",".join(
                [
                    lab["name"]
                    for lab in labels
                    if lab["id"] in [l['id'] for l in n["labels"]]
                ]
            )
            json_articles.append(new)
            n_new += 1
        json.dump(
            json_articles,
            json_file_to_update.open("w"),
            ensure_ascii=False,
            indent=2,
        )
    tqdm.write(f"Done updating {json_file_to_update}!")

@typechecked
def review(
    json_file_to_update: Union[str, PosixPath],
    omnivore_api_key: Optional[str] = None,
    ):
    if omnivore_api_key is None:
        omnivore_api_key = _load_api_key()
    log.info("Starting omnivore review")
    try:
        client = OmnivoreQL(omnivore_api_key)
        labels = client.get_labels()["labels"]["labels"]
    except Exception as err:
        raise Exception(f"Error when logging to OmnivoreQL then loading labels: '{err}'")

    json_file_to_update =  Path(json_file_to_update)
    assert json_file_to_update.exists()
    try:
        json_articles = json.load(Path(json_file_to_update).open("r"))
    except Exception as err:
        raise Exception(f"Error when loading {json_file_to_update}: '{err}'")
    assert isinstance(json_articles, list), f"loaded json is not a list"
    assert all(isinstance(article, dict) for article in json_articles), f"loaded json is not a list of dict"

    @typechecked
    def update_labels(instance: mini_LiTOY, entry1: dict, entry2: dict) -> None:
        # TODO: add ELO label
        breakpoint()

    log.info("Starting mini LiTOY")
    mini_litoy = mini_LiTOY(
        output_json=json_file_to_update,
        callback=update_labels,
    )


if __name__== "__main__":
    fire.Fire()
