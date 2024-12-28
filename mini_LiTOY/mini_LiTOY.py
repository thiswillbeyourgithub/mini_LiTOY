import copy
import sys
import os
import logging
import json
import time
from pathlib import Path, PosixPath
from typing import Optional, Callable, Union
import io

from platformdirs import user_log_dir, user_config_dir, user_cache_dir
import fire
from typeguard import typechecked
from rich.markdown import Markdown
from rich.console import Console
from rich.table import Table
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import clear
import rtoml as toml


class LockedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locked = True

    def __setitem__(self, key, value):
        if self._locked and key not in self:
            raise KeyError(f"Cannot create new key '{key}'. Only existing keys can be modified.")
        super().__setitem__(key, value)

    def __deepcopy__(self, memo):
        new_instance = LockedDict(self)
        assert id(new_instance) != id(self)
        return new_instance


class mini_LiTOY:
    VERSION: str = "0.1.5"
    inertia_values = [30, 25, 20, 15, 10]
    questions = [
        "Which is more important to you?'",
        "Which takes the less time?",
    ]
    ELO_norm = 40
    ELO_default = 100

    LockedDict = LockedDict
    default_subdict = dict(
        {
        "q_n_comparison": 0,
        "q_ELO": ELO_default,
        }
    )
    default_dict = LockedDict(
        {
        "entry": None,
        "g_n_comparison": 0,
        "g_ELO": ELO_default,
        "all_ELO": {},
        "id": None,
        "metadata": {},
        }
    )
    for q in questions:
        default_dict["all_ELO"][q] = copy.deepcopy(default_subdict)

    @typechecked
    def __init__(
        self,
        input_file: Optional[Union[PosixPath, str, io.TextIOWrapper]] = None,
        output_file: Optional[Union[PosixPath, str, io.TextIOWrapper]] = None,
        callback: Optional[Callable] = None,
        verbose: bool = False,
        ):
        """
        Parameters
        ----------

        :param input_file: Path a txt file, parsed as one entry per line, ignoring empty lines and those starting with #.
        :param output_file: Path the file file that will be updated as ELOs get updated.
        :param callback: Callable, will be called just after updating the json file. This is intended for use when imported. See examples folder.
        :param verbose: bool, increase verbosity
        """
        self.verbose = verbose
        self.p(f"Logdir: {log_dir}")
        self.p(f"recovery_dir: {recovery_dir}")
        self.p(f"Initializing mini_LiTOY with input_file={input_file}, output_file={output_file}")
        self.recovery_file = recovery_dir / str(int(time.time()))

        if not input_file and not output_file:
            log.error("Either input_file or output_file must be provided")
            raise ValueError("Either input_file or output_file must be provided")

        if not output_file:
            log.error("output_file must be provided")
            raise ValueError("output_file must be provided")
        output_format = Path(output_file).suffix
        assert output_format in ["json", "toml"], f"output_file extension must be 'json' or 'toml' not '{output_format}'"

        self.output_file = output_file
        self.callback = callback
        self.output_format = output_format

        # load previous data
        self.lines = []
        if self.output_file and Path(self.output_file).exists():
            self.p(f"Loading data from {self.output_file}")
            with open(self.output_json, 'r') as file:
                if output_format == "json":
                    data = json.load(file)
                elif output_format == "toml":
                    data = toml.load(file)
                else:
                    raise ValueError(f"output_format must be 'json' or 'toml', not '{output_format}'")
                assert isinstance(data, list) and all(isinstance(item, dict) for item in data), "Output file must be a list of dictionaries"
            self.alldata = [self.LockedDict(di) for di in data]
        else:
            self.alldata = []

        # check validity of data
        for ie, entry in enumerate(self.alldata):
            for k in entry.keys():
                assert k in [
                    "entry","g_n_comparison", "g_ELO", "all_ELO", "id", "metadata",
                ], f"Unexpected key {k} in entry #{ie}"
            for k in ["entry","g_n_comparison", "g_ELO", "all_ELO", "id", "metadata"]:
                assert k in entry.keys(), (
                    f"Entry #{ie} missing key {k}:\n{entry}"
                )
            for q in self.questions:
                if q not in entry["all_ELO"]:
                    entry["all_ELO"][q] = LockedDict({"q_n_comparison": 0, "q_ELO": self.ELO_default})

            for q, d in entry["all_ELO"].items():
                assert isinstance(d, dict)
                assert q in self.questions, f"Entry #{ie} contains a question that is not part of self.questions: '{q}'"
                if not isinstance(d, self.LockedDict):
                    entry["all_ELO"][q] = self.LockedDict(d)
                    d = self.LockedDict(d)
                assert isinstance(d, self.LockedDict)
                assert isinstance(q, str), f"questions in entry #{ie} entry['all_ELO'] must be strings not {type(q)} '({q})'"
                assert all(dd in ["q_n_comparison", "q_ELO"] for dd in  d.keys()), f"Invalid entry['all_ELO']: {q}:{d} in entry #{ie}"
                assert all(k in d.keys() for k in ["q_n_comparison", "q_ELO"]), f"Invalid entry['all_ELO']: {q}:{d} in entry #{ie}"

            # all question of the dict must be part of self.questions, but the dict can have fewer, in that case they will be created here
            for q in entry["all_ELO"].keys():
                assert q in self.questions, f"Entry #{ie} contains a question that is not part of self.questions: '{q}'"

        # check no entries have the same name or id
        texts = [ent["entry"] for ent in self.alldata]
        ids = [ent["id"] for ent in self.alldata]
        dup_t = set()
        dup_i = set()
        for t in texts:
            if texts.count(t) > 1:
                dup_t.add(t)
        for i in ids:
            if ids.count(i) > 1:
                dup_i.add(i)
        if dup_t:
            self.p(f"Found {len(dup_t)} entries whose 'entry' text is identical:", type="error")
            for t in dup_t:
                self.p(t, type="error")
        if dup_i:
            self.p(f"Found {len(dup_i)} entries whose 'id' value is identical:", type="error")
            for i in dup_i:
                self.p(i, type="error")
        # if dup_t or dup_i:
        #     raise Exception()

        if self.alldata:
            max_id = max(
                [
                    int(item["id"])
                    if str(item["id"]).isdigit()
                    else it
                    for it, item in enumerate(self.alldata)
                ]
            )
        else:
            max_id = 1

        if input_file:
            self.p(f"Reading input from {input_file}")
            with open(input_file, 'r') as file:
                input_content = file.read()
            for line in input_content.splitlines():
                line = line.strip()
                if (not line) or line.startswith("#"):
                    continue
                if not any(entry["entry"] == line for entry in self.alldata):
                    max_id += 1
                    entry = copy(self.default_dict)
                    entry["entry"] = line
                    entry["id"] = max_id
                    self.alldata.append(entry)

        self.console = Console()
        self.p("Starting comparison loop")
        self.run_comparison_loop()

    @typechecked
    def run_comparison_loop(self) -> None:
        counter = 0
        while True:
            if not self.verbose:
                clear()  # clearing screen seems to hide the debug prints
            self.p("Picking two entries for comparison")
            entry1, entry2 = self.pick_two_entries()
            assert entry1 in self.alldata and entry2 in self.alldata
            self.p(f"Displaying comparison table for entries {entry1['id']} and {entry2['id']}")
            self.display_comparison_table(entry1, entry2)
            bindings = KeyBindings()

            self.skip = False

            @bindings.add(Keys.Enter)
            @bindings.add(' ')
            @bindings.add('s')
            @bindings.add('1')
            @bindings.add('2')
            @bindings.add('3')
            @bindings.add('4')
            @bindings.add('5')
            @bindings.add('a')
            @bindings.add('z')
            @bindings.add('e')
            @bindings.add('r')
            @bindings.add('t')
            def _(event):
                key = event.key_sequence[0].key
                if key is Keys.Enter:
                    key = event.app.current_buffer.text.strip()
                if key == "s" or key == " ":
                    self.skip = True
                elif key in 'azert':
                    key = str('azert'.index(key) + 1)
                event.app.exit(result=key)

            answers = {}
            for iq, q in enumerate(self.questions):
                if len(self.questions) > 1:
                    prefix = f"[{iq+1}/{len(self.questions)+1}] "
                else:
                    prefix = ""

                try:
                    answers[q] = prompt(f"{prefix}{q} (1-5 or a-z-e-r-t and s or ' ' to skip): ", key_bindings=bindings)
                except (KeyboardInterrupt, EOFError):
                    self.p("Exiting due to keyboard interrupt")
                    raise SystemExit("\nExiting. Goodbye!")

                self.p(f"User selected answer: '{answers[q]}'")
                if self.skip:
                    self.p("Skipping this comparison")
                    continue
                assert answers[q].isdigit(), f"Answer should be an int"
                answers[q] = int(answers[q])

                n_comparison_1 = entry1["all_ELO"][q]["q_n_comparison"]
                K1 = self.inertia_values[n_comparison_1] if n_comparison_1 <= len(self.inertia_values) else self.inertia_values[-1]
                n_comparison_2 = entry2["all_ELO"][q]["q_n_comparison"]
                K2 = self.inertia_values[n_comparison_2] if n_comparison_2 <= len(self.inertia_values) else self.inertia_values[-1]

                new_elo1, new_elo2 = self.update_elo(answers[q], entry1["all_ELO"][q]["q_ELO"], entry2["all_ELO"][q]["q_ELO"], K1, K2)
                entry1["all_ELO"][q]["q_ELO"], entry2["all_ELO"][q]["q_ELO"] = new_elo1, new_elo2
                self.p(f"Updated ELOs: entry1={new_elo1}, entry2={new_elo2}")

                entry1["all_ELO"][q]["q_n_comparison"] += 1
                entry2["all_ELO"][q]["q_n_comparison"] += 1

                entry1["g_ELO"] = int(sum([entry1["all_ELO"][_q]["q_ELO"] for _q in entry1["all_ELO"].keys()]) / len(self.questions))
                entry2["g_ELO"] = int(sum([entry2["all_ELO"][_q]["q_ELO"] for _q in entry2["all_ELO"].keys()]) / len(self.questions))
                entry1["g_n_comparison"] = sum([entry1["all_ELO"][_q]["q_n_comparison"] for _q in entry1["all_ELO"].keys()])
                entry2["g_n_comparison"] = sum([entry2["all_ELO"][_q]["q_n_comparison"] for _q in entry2["all_ELO"].keys()])

                assert entry1 in self.alldata and entry2 in self.alldata

                self.store_data()
                self.p("Stored data to file")

                counter += 1

                if self.callback is not None:
                    self.p("Calling callback")
                    self.callback(
                        self,
                        entry1,
                        entry2,
                    )

    @typechecked
    def display_comparison_table(self, entry1: LockedDict, entry2: LockedDict) -> None:
        terminal_width = os.get_terminal_size().columns

        table = Table(title="Comparison", width=terminal_width)

        table.add_column("Entries", justify="center", no_wrap=True, width=terminal_width//5)
        table.add_column(str(entry1['id']), justify="center", width = terminal_width//5*2)
        table.add_column(str(entry2['id']), justify="center", width = terminal_width//5*2)


        table.add_row("[bold]Content", "[bold]" + str(entry1["entry"]), "[bold]" + str(entry2["entry"]))
        table.add_row("", "", "")
        table.add_row("", "", "")
        table.add_row("[bold]Nb compar", str(entry1["g_n_comparison"]), str(entry2["g_n_comparison"]))
        table.add_row("[bold]Global ELO", str(entry1["g_ELO"]), str(entry2["g_ELO"]))
        for q in self.questions:
            table.add_row(f"[bold]{q[:10]}...", str(entry1["all_ELO"][q]["q_ELO"]), str(entry2["all_ELO"][q]["q_ELO"]))

        metadata_keys = []
        if entry1["metadata"]:
            [metadata_keys.append(k) for k in entry1["metadata"].keys()]
        if entry2["metadata"]:
            for k in entry2["metadata"]:
                if k not in metadata_keys:
                    metadata_keys.append(k)
        if metadata_keys:
            table.add_row("[bold]Metadata", "", "")
        for mk in metadata_keys:
            if mk in entry1["metadata"]:
                val1 = entry1["metadata"][mk]
            else:
                val1 = ""
            if isinstance(val1, dict):
                val1 = json.dumps(val1)
            if mk in entry2["metadata"]:
                val2 = entry2["metadata"][mk]
            else:
                val2 = ""
            if isinstance(val2, dict):
                val2 = json.dumps(val2)
            val1 = str(val1)
            val2 = str(val2)
            table.add_row(mk.title(), val1, val2)

        self.console.print(table)

    @typechecked
    def update_elo(self, answer: int, elo1: int, elo2: int, k1: int, k2: int) -> tuple[int, int]:
        """
        Update ELO scores based on the answer.

        :param answer: int, number of wins for the first player (1-5)
        :param elo1: int, ELO score of the first player
        :param elo2: int, ELO score of the second player
        :param k1: int, K value of the first player
        :param k2: int, K value of the second player

        :return: tuple, updated ELO scores (new_elo1, new_elo2)
        """
        if not (1 <= answer <= 5):
            raise ValueError("Answer must be a digit between 1 and 5")

        expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / self.ELO_norm))
        assert expected_score1 < 1 and expected_score1 > 0
        expected_score2 = 1 - expected_score1

        actual_score1 = (5 - answer) / 5
        actual_score2 = 1 - actual_score1

        new_elo1 = elo1 + k1 * (actual_score1 - expected_score1)
        new_elo2 = elo2 + k2 * (actual_score2 - expected_score2)

        return round(new_elo1), round(new_elo2)

    @typechecked
    def pick_two_entries(self) -> tuple[LockedDict, LockedDict]:
        """
        Pick three entries at random, then return the first of the three and the one with the lowest n_comparison between the other two.

        :return: tuple, two entries as LockedDict
        """
        import random

        if len(self.alldata) < 5:
            raise ValueError("You need at least 5 entries to start comparing")

        entries_nb = random.sample(range(len(self.alldata)), 3)

        entry1 = self.alldata[entries_nb[0]]
        entry2 = self.alldata[entries_nb[1]]
        entry3 = self.alldata[entries_nb[2]]

        assert entry2 != entry1 and entry3 != entry1
        assert all(entr in self.alldata for entr in [entry1, entry2, entry3])
        if entry2["g_n_comparison"] <= entry3["g_n_comparison"]:
            return entry1, entry2
        else:
            return entry1, entry3

    @typechecked
    def store_data(self) -> None:
        if not hasattr(self, 'output_file') or not self.output_file:
            raise AttributeError("Missing attribute: 'output_file'")
        if not hasattr(self, 'alldata') or not isinstance(self.alldata, list):
            raise AttributeError("Missing or invalid attribute: 'alldata'")

        # always save to recovery file
        with open(self.recovery_file, 'w', encoding='utf-8') as file:
            if self.output_format == "json":
                json.dump(self.alldata, file, ensure_ascii=False, indent=4)
            elif self.output_format == "toml":
                toml.dump(self.alldata, file, pretty=True)
            else:
                raise ValueError(self.output_format)

        with open(self.output_json, 'w', encoding='utf-8') as file:
            if self.output_format == "json":
                json.dump(self.alldata, file, ensure_ascii=False, indent=4)
            elif self.output_format == "toml":
                toml.dump(self.alldata, file, pretty=True)
            else:
                raise ValueError(self.output_format)

    @typechecked
    def p(self, message: str, type: str = "info") -> str:
        "printer"
        if type == "info":
            log.info(message)
            if self.verbose:
                print(message)
        elif type == "error":
            log.error(message)
            erro(message)
        else:
            raise ValueError(type)
        return message

mini_LiTOY.__doc__ = mini_LiTOY.__init__.__doc__

console = Console()

@typechecked
def erro(message: str) -> str:
    "print error message"
    md = Markdown(message)
    console.print(md, style="red")
    return message


# figure out logging dir
try:
    log_dir = user_log_dir(
        appname="mini_LiTOY",
        version=mini_LiTOY.VERSION,
    )
    par = Path(log_dir).parent.parent.parent
    assert par.exists() and par.is_dir()
except Exception as err:
    try:
        log_dir = user_config_dir(
            appname="mini_LiTOY",
            version=mini_LiTOY.VERSION,
        )
        par = Path(log_dir).parent.parent.parent
        assert par.exists() and par.is_dir()
    except Exception as err2:
        raise Exception(erro(f"# Errors when trying to find an appropriate log dir:\n- '{err}'\n- '{err2}'"))

log_dir = Path(log_dir)
par = Path(log_dir).parent.parent.parent
assert par.exists() and par.is_dir()
log_dir.mkdir(parents=True, exist_ok=True)
log_file = str((log_dir / "mini_litoy.logs.txt").resolve().absolute())

# Configure logging
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

cache_dir = Path(
        user_cache_dir(
            appname="mini_LiTOY",
            version=mini_LiTOY.VERSION,
        )
)
par = cache_dir.parent.parent
assert par.exists() and par.is_dir()
recovery_dir = cache_dir / "recovery"
recovery_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    fire.Fire(mini_LiTOY)
