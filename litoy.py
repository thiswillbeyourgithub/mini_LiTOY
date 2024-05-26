import fire
import json
import os
from rich.console import Console
from rich.table import Table

class LiTOY:
    def __init__(self, input_file=None, json_file=None, question="most important?"):
        self.question = question
        self.lines = []
        if json_file and os.path.exists(json_file):
            with open(json_file, 'r') as file:
                data = json.load(file)
                assert isinstance(data, list) and all(isinstance(item, dict) for item in data), "JSON file must be a list of dictionaries"
            self.json_data = data
        else:
            self.json_data = []

        max_id = max((item.get("id", 0) for item in self.json_data), default=0)


        if input_file:
            with open(input_file, 'r') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line and not any(entry["entry"] == stripped_line for entry in self.json_data):
                        max_id += 1
                        entry = {
                            "entry": stripped_line,
                            "K": 32,
                            "ELO": 1000,  # Sensible default ELO
                            "id": max_id
                        }
                        self.json_data.append(entry)

        self.console = Console()
        self.run_comparison_loop()

    def run_comparison_loop(self):
        counter = 0
        while True:
            entry1, entry2 = self.pick_two_entries()
            self.display_comparison_table(entry1, entry2)
            while True:
                try:
                    answer = int(self.console.input("[bold yellow]Which entry do you prefer? (1-5): [/bold yellow]"))
                    if 1 <= answer <= 5:
                        break
                    else:
                        self.console.print("[bold red]Invalid input. Please enter a number between 1 and 5.[/bold red]")
                except ValueError:
                    self.console.print("[bold red]Invalid input. Please enter a number between 1 and 5.[/bold red]")

            new_elo1, new_elo2 = self.update_elo(answer, entry1["ELO"], entry2["ELO"], entry1["K"])
            entry1["ELO"], entry2["ELO"] = new_elo1, new_elo2

            # Update K values (example logic, can be adjusted)
            entry1["K"] = max(1, entry1["K"] - 10)
            entry2["K"] = max(1, entry2["K"] - 10)

            self.store_json_data()

            counter += 1

    def display_comparison_table(self, entry1, entry2):
        table = Table(title="Comparison")

        table.add_column("ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Entry", justify="center", style="magenta")
        table.add_column("K", justify="center", style="green")
        table.add_column("ELO", justify="center", style="red")

        table.add_row(str(entry1["id"]), entry1["entry"], str(entry1["K"]), str(entry1["ELO"]))
        table.add_row(str(entry2["id"]), entry2["entry"], str(entry2["K"]), str(entry2["ELO"]))

        self.console.print(table)

    def update_elo(self, answer, elo1, elo2, k):
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

    def pick_two_entries(self):
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
    def store_json_data(self):
        if not hasattr(self, 'json_file') or not self.json_file:
            raise AttributeError("Missing attribute: 'json_file'")
        if not hasattr(self, 'json_data') or not isinstance(self.json_data, list):
            raise AttributeError("Missing or invalid attribute: 'json_data'")

        with open(self.json_file, 'w', encoding='utf-8') as file:
            json.dump(self.json_data, file, ensure_ascii=False, indent=4)

fire.Fire(LiTOY)
