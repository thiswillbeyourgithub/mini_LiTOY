import fire
import json
import os
import logging
from rich.console import Console
from rich.table import Table

from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger()

class LiTOY:
    def __init__(self, input_file: str = None, json_file: str = None, question: str = "most important?", mode: str = "review"):
        log.info("Initializing LiTOY with input_file=%s, json_file=%s, question=%s, mode=%s", input_file, json_file, question, mode)
        if not input_file and not json_file:
            log.error("Either input_file or json_file must be provided")
            raise ValueError("Either input_file or json_file must be provided")

        if not json_file:
            log.error("json_file must be provided")
            raise ValueError("json_file must be provided")

        self.json_file = json_file
        self.question = question
        self.mode = mode
        self.lines = []
        if json_file and os.path.exists(json_file):
            log.info("Loading data from %s", json_file)
            with open(json_file, 'r') as file:
                data = json.load(file)
                assert isinstance(data, list) and all(isinstance(item, dict) for item in data), "JSON file must be a list of dictionaries"
            self.json_data = data
        else:
            self.json_data = []

        max_id = max((item.get("id", 0) for item in self.json_data), default=0)


        if input_file:
            log.info("Reading input from %s", input_file)
            with open(input_file, 'r') as file:
                for line in file:
                    stripped_line = line.lstrip('-#').strip()
                    if stripped_line and not any(entry["entry"] == stripped_line for entry in self.json_data):
                        max_id += 1
                        entry = {
                            "entry": stripped_line,
                            "K": 30,
                            "ELO": 1000,  # Sensible default ELO
                            "id": max_id
                        }
                        self.json_data.append(entry)

        self.console = Console()
        if self.mode == "review":
            log.info("Starting comparison loop")
            self.run_comparison_loop()
        elif self.mode == "export":
            log.info("Exporting to markdown")
            self.export_to_markdown()
        else:
            log.error("Invalid mode: %s", self.mode)
            raise ValueError("Invalid mode. Use 'review' or 'export'.")

    def run_comparison_loop(self) -> None:
        counter = 0
        try:
            while True:
                log.info("Picking two entries for comparison")
                entry1, entry2 = self.pick_two_entries()
                log.info("Displaying comparison table for entries %d and %d", entry1["id"], entry2["id"])
                self.display_comparison_table(entry1, entry2)
                bindings = KeyBindings()

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
                    if key in 'azert':
                        key = str('azert'.index(key) + 1)
                    event.app.exit(result=key)

                answer = prompt(f"{self.question} (1-5 or a-z-e-r-t): ", key_bindings=bindings)
                log.info("User selected answer: %s", answer)
                if answer in 'azert':
                    answer = str('azert'.index(answer) + 1)
                answer = int(answer)

                new_elo1, new_elo2 = self.update_elo(answer, entry1["ELO"], entry2["ELO"], entry1["K"])
                entry1["ELO"], entry2["ELO"] = new_elo1, new_elo2
                log.info("Updated ELOs: entry1=%d, entry2=%d", new_elo1, new_elo2)

                # Update K values (example logic, can be adjusted)
                entry1["K"] = max(10, entry1["K"] - 5)
                entry2["K"] = max(10, entry2["K"] - 5)

                self.store_json_data()
                log.info("Stored JSON data")

                counter += 1
        except KeyboardInterrupt:
            log.info("Exiting due to keyboard interrupt")
            raise SystemExit("\nExiting. Goodbye!")

    def display_comparison_table(self, entry1: dict, entry2: dict) -> None:
        import os
        terminal_width = os.get_terminal_size().columns
        table = Table(title="Comparison", width=terminal_width)

        table.add_column("ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Entry", justify="left", style="magenta")
        table.add_column("K", justify="center", style="green")
        table.add_column("ELO", justify="center", style="red")

        table.add_row(str(entry1["id"]), entry1["entry"], str(entry1["K"]), str(entry1["ELO"]))
        table.add_row(str(entry2["id"]), entry2["entry"], str(entry2["K"]), str(entry2["ELO"]))

        self.console.print(table)

    def update_elo(self, answer: int, elo1: int, elo2: int, k: int) -> tuple[int, int]:
        """
        Update ELO scores based on the answer.
        
        :param answer: int, number of wins for the first player (1-5)
        :param elo1: int, ELO score of the first player
        :param elo2: int, ELO score of the second player
        :return: tuple, updated ELO scores (new_elo1, new_elo2)
        """
        if not (1 <= answer <= 5):
            raise ValueError("Answer must be a digit between 1 and 5")

        expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        expected_score2 = 1 - expected_score1

        actual_score1 = answer / 5
        actual_score2 = 1 - actual_score1

        new_elo1 = elo1 + k * (actual_score1 - expected_score1)
        new_elo2 = elo2 + k * (actual_score2 - expected_score2)

        return round(new_elo1), round(new_elo2)

    def pick_two_entries(self) -> tuple[dict, dict]:
        """
        Pick three entries at random, then return the first of the three and the one with the highest K among the other two.
        
        :return: tuple, two entries as dictionaries
        """
        import random

        if len(self.json_data) < 2:
            raise ValueError("Not enough entries to pick from")

        entries = random.sample(self.json_data, 3)

        entry1 = entries[0]
        entry2 = entries[1]
        entry3 = entries[2]

        if entry2["K"] < entry3["K"]:
            entry2 = entry3

        return entry1, entry2
    def store_json_data(self) -> None:
        if not hasattr(self, 'json_file') or not self.json_file:
            raise AttributeError("Missing attribute: 'json_file'")
        if not hasattr(self, 'json_data') or not isinstance(self.json_data, list):
            raise AttributeError("Missing or invalid attribute: 'json_data'")

        with open(self.json_file, 'w', encoding='utf-8') as file:
            json.dump(self.json_data, file, ensure_ascii=False, indent=4)

    def export_to_markdown(self) -> None:
        sorted_entries = sorted(self.json_data, key=lambda x: x["ELO"], reverse=True)
        markdown_lines = [f"- {entry['entry']}" for entry in sorted_entries]

        markdown_file = self.json_file.replace('.json', '.md')
        log.info("Exporting to markdown file: %s", markdown_file)
        if os.path.exists(markdown_file):
            confirm = input(f"{markdown_file} already exists. Do you want to overwrite it? (y/n): ")
            if confirm.lower() != 'y':
                log.info("Export cancelled by user")
                print("Export cancelled.")
                return

        with open(markdown_file, 'w', encoding='utf-8') as file:
            file.write("\n".join(markdown_lines))

        log.info("Exported to %s", markdown_file)
        print(f"Exported to {markdown_file}")

if __name__ == "__main__":
    fire.Fire(LiTOY)
