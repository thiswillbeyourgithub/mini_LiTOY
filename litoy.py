import fire
import json
import os

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

        counter = 0
        while counter < 10:  # Arbitrary counter limit to prevent infinite loops
            entry1, entry2 = self.pick_two_entries()
            counter += 1

        if input_file:
            with open(input_file, 'r') as file:
                for line in file:
                    stripped_line = line.strip()
                    if stripped_line and not any(entry["entry"] == stripped_line for entry in self.json_data):
                        max_id += 1
                        entry = {
                            "entry": stripped_line,
                            "K": 40,
                            "ELO": 1000,  # Sensible default ELO
                            "id": max_id
                        }
                        self.json_data.append(entry)

    def update_elo(self, answer, elo1, elo2):
        """
        Update ELO scores based on the answer.
        
        :param answer: int, number of wins for the first player (1-5)
        :param elo1: int, ELO score of the first player
        :param elo2: int, ELO score of the second player
        :return: tuple, updated ELO scores (new_elo1, new_elo2)
        """
        if not (1 <= answer <= 5):
            raise ValueError("Answer must be a digit between 1 and 5")

        k = 32  # K-factor, can be adjusted
        expected_score1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        expected_score2 = 1 - expected_score1

        actual_score1 = answer / 5
        actual_score2 = 1 - actual_score1

        new_elo1 = elo1 + k * (actual_score1 - expected_score1)
        new_elo2 = elo2 + k * (actual_score2 - expected_score2)

        return round(new_elo1), round(new_elo2)

    def pick_two_entries(self):
        """
        Pick two entries at random, check the one with the highest K, and return two entries as a tuple of two dicts.
        
        :return: tuple, two entries as dictionaries
        """
        import random

        if len(self.json_data) < 2:
            raise ValueError("Not enough entries to pick from")

        entry1 = random.choice(self.json_data)
        entry2 = random.choice(self.json_data)
        while entry2 == entry1:
            entry2 = random.choice(self.json_data)

        if entry1["K"] < entry2["K"]:
            entry1, entry2 = entry2, entry1

        return entry1, entry2
    fire.Fire(LiTOY)
