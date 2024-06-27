import time
from pathlib import Path, PosixPath
import json
from typing import List, Union, Optional
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

# Configure logging
logging.basicConfig(filename='omnivore_logs.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

from mini_LiTOY import mini_LiTOY

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

default_dict = {
        "n_comparison": 0,
        "ELO": mini_LiTOY.ELO_default,
        "metadata": {},
}
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
def exec_query(base_query: str, d1: str, d2: str, omnivore_api_key: str) -> List:
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
    return d

@typechecked
def update_js(
    json_file_to_update: Union[str, PosixPath],
    omnivore_api_key: Optional[str] = None,
    start_date: str = "2023-04-01",
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
    d = datetime.today()
    end_date = f"{d.year}-{d.month}-{d.day}"  + relativedelta(days=1)
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    current_date = start_date
    dates = []
    while current_date <= end_date:
        newdate = [
                (current_date - relativedelta(days=1)).strftime("%Y-%m-%d"),
                (current_date + relativedelta(days=days_diff)).strftime("%Y-%m-%d"),
        ]
        current_date += relativedelta(days=days_diff)
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

    present_ids = [article["id"] for article in json_articles]

    # execute async queries
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
                )
                for d1, d2 in tqdm(dates, desc="Querying", unit="date_ranges")
            ]
            results = await asyncio.gather(*tasks)
        return results
    results = asyncio.run(main())

    assert len(results) == len(dates), f"Number of results={len(results)} but number of date ranges: {len(dates)}"

    for idate, d1, d2 in tqdm(enumerate(dates), total=len(dates)):
        d = results[idate]
        edges = d["search"]["edges"]

        if len(edges) >= 100:
            raise Exception(f"Found {len(edges)} articles for date '{d1}..{d2}', this is above 100 so use a lower time_window.")

        n_new = 0
        for e in edges:
            n = e["node"]
            if n["id"] in present_ids:
                continue

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
