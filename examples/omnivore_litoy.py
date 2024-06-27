import copy
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
from omnivoreql import OmnivoreQL, CreateLabelInput
from mini_LiTOY import mini_LiTOY

# Configure logging
logging.basicConfig(
    filename='omnivore_logs.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger()


MAX_REQUEST_TRIALS = 5
MAX_CONCURRENCY = 10
DUPLICATE_LABEL = "litoy_duplicates"

@typechecked
def _load_api_key() -> str:
    if "OMNIVORE_API_KEY" not in os.environ:
        raise Exception("No OMNIVORE_API_KEY found in environment, and not given as arugment")
    elif not os.environ["OMNIVORE_API_KEY"]:
        raise Exception("Empty OMNIVORE_API_KEY found in environment")
    omnivore_api_key = os.environ["OMNIVORE_API_KEY"]
    return omnivore_api_key

default_dict = copy.deepcopy(mini_LiTOY.default_dict)
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
def label_input(
    name: str,
    color: str = "#ff0000",  # red
    description: str = "Label created by mini_litoy/examples/omnivore_litoy.py",
    ) -> CreateLabelInput:
    return CreateLabelInput(
        name,
        color,
        description,
    )

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
    base_query: str = "in:inbox -type:highlights sort:saved saved:$start_date..$end_date -in:archive -label:" + DUPLICATE_LABEL,
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
    pbar.close()

    assert len(results) == len(dates), f"Number of results={len(results)} but number of date ranges: {len(dates)}"

    present_ids = [article["id"] for article in json_articles]
    extra_ids = copy.deepcopy(present_ids)

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
                if n["id"] in extra_ids:
                    extra_ids.remove(n["id"])
                continue
            present_ids.append(n["id"])

            new = default_dict.copy()
            new["id"] = n["id"]
            new["entry"] = n['title']
            if n["author"] and n["siteName"]:
                if n["author"] in n["siteName"]:
                    new["entry"] += f"\non {n['siteName']}"
                elif n["siteName"] in n["author"]:
                    new["entry"] += f"\nby {n['author']}"
            else:
                if n["author"] and n["author"] not in new["entry"]:
                    new["entry"] += f"\nby {n['author']}"
                if n["siteName"] and n["siteName"] not in new["entry"]:
                    new["entry"] += f"\non {n['siteName']}"
            new["metadata"] = {k: n[k] for k in metadata_keys}
            # don't save labels as dict but as a list
            new["metadata"]["labels"] = [
                lab["name"]
                for lab in labels
                if lab["id"] in [l['id'] for l in n["labels"]]
            ]
            json_articles.append(new)
            n_new += 1

    # check no entries have the same name or id
    ids = [ent["id"] for ent in json_articles]
    dup_i = set()
    for i in ids:
        if ids.count(i) > 1:
            dup_i.add(i)
    if dup_i:
        log.info(f"Found {len(dup_i)} entries whose 'id' value is identical:")
        for i in dup_i:
            log.info(f"- {i}")
        raise Exception()

    texts = [ent["entry"] for ent in json_articles]
    dup_t = set()
    dup_ind = []
    for ind, t in enumerate(texts):
        if texts.count(t) > 1:
            dup_t.add(t)
            dup_ind.append(ind)
    if dup_t:
        log.info(f"Found {len(dup_t)} entries whose 'entry' text is identical:")
        for t in dup_t:
            log.info(f"- {t}")

        # create label if missing
        if DUPLICATE_LABEL not in [lab["name"] for lab in labels]:
            log.info(f'Creating label "{DUPLICATE_LABEL}"')
            client.create_label(label_input(name=DUPLICATE_LABEL))
            labels = client.get_labels()["labels"]["labels"]
        assert DUPLICATE_LABEL in [lab["name"] for lab in labels], f"Failed to create label {DUPLICATE_LABEL}"
        dup_lab_id = [lab["id"] for lab in labels if lab["name"] == DUPLICATE_LABEL]

        for ind in dup_ind:
            article = json_articles[ind]
            page_id=article["id"]
            old_lab_ids = [lab["id"] for lab in labels if lab["name"] in article["metadata"]["labels"]]
            log.info(f"Adding duplicate label to article with id '{page_id}'")
            client.set_page_labels_by_ids(
                page_id=json_articles[ind]["id"],
                label_ids=dup_lab_id + old_lab_ids,
            )
            while article in json_articles:
                json_articles.remove(article)

    # remove articles that were archived
    if extra_ids:
        log.info(f"Found {len(extra_ids)} articles that are in the local file"
                 "but not in the server output. Removing those ids from the "
                "local file. Will be removed:")
        for art_id in extra_ids:
            article = [art for art in json_articles if art["id"] == art_id]
            assert len(article ) == 1
            article = article[0]
            entry = article["entry"]
            log.info(f"- {entry}")
            while article in json_articles:
                json_articles.remove(article)

    assert json_articles, "No article left!"
    json.dump(
        json_articles,
        json_file_to_update.open("w"),
        ensure_ascii=False,
        indent=2,
    )
    log.info(f"Done updating {json_file_to_update}!")

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
        all_labels = client.get_labels()["labels"]["labels"]
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

    all_labels = [all_labels]

    @typechecked
    def update_labels(
        instance: mini_LiTOY,
        entry1: dict,
        entry2: dict,
        client: OmnivoreQL = client,
        all_labels: List[List] = all_labels,
        ) -> None:
        for entr in [entry1, entry2]:
            entr_labels = entr["metadata"]["labels"]
            score = str(int(entr["g_ELO"] / 10))
            new_lab = f"litoy_{score}"

            # label not changed
            if new_lab in entr_labels:
                log.info(f"Entry with id {entr['id']} already has the label {new_lab}")
                continue

            # create label if needed
            if new_lab not in [lab["name"] for lab in all_labels[0]]:
                log.info(f"Entry with id {entr['id']}: creating label {new_lab}")
                client.create_label(label_input(name=new_lab))
                all_labels[0] = client.get_labels()["labels"]["labels"]

            # add the label
            old_lab_ids = [
                lab["id"] for lab in all_labels[0]
                if lab["name"] in entr["metadata"]["labels"] and not lab["name"].startswith("litoy_")
            ]
            new_lab_id = [
                lab["id"]
                for lab in all_labels[0]
                if lab["name"] == new_lab
            ]
            assert len(new_lab_id) == 1
            log.info(f"Entry with id {entr['id']}: setting labels to {new_lab_id + old_lab_ids}")
            client.set_page_labels_by_ids(
                page_id=entr["id"],
                label_ids=new_lab_id + old_lab_ids,
            )

    log.info("Starting mini LiTOY")
    mini_litoy = mini_LiTOY(
        output_json=json_file_to_update,
        callback=update_labels,
        verbose=True,
    )


if __name__== "__main__":
    fire.Fire()
