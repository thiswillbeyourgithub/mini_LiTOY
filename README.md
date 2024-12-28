# mini_LiTOY
Minimalist approach to the [LiTOY task sorting algorithm](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You) based on [ELO scores](https://en.wikipedia.org/wiki/Elo_rating_system).

# Features
* Minimalist
* Statically typed via typeguard
* Made to be extensible
* Supports callbacks

# FAQ

### What is LiTOY?
[LiTOY](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You) was a personal project. The idea is to dump all your TODOs in one place, then rank them using the mean ELO scores of question answer. The default questions are `Which is more important to you?` and `Which takes the less time?` but you can use anything you like instead!

### Why make mini_LiTOY?
mini_LiTOY's idea is to keep the code idea but this time in a minimalist python script. As long as the user takes care of storing the tasks in a text files, the LiTOY algorithm will update an output json (or toml) based on the score. The first use for this will be to rank my [Omnivore](https://github.com/omnivore-app/omnivore) reading queue by downloading titles of articles to read via their API, then uploading the ELO score as a label. This can be found in the examples folder.

# Usage
* `python -m pip install mini_LiTOY` or `python -m pip install git+https://github.com/thiswillbeyourgithub/mini_LiTOY.git`
* Then you can either launch it with `python -m mini_LiTOY [ARGS]` or use the alias  `mlitoy [ARGS]`
* `mlitoy --input_file my_text_file.txt --output_path output_file.json`
## Notes
* each new (nonempty nor commented) line in input_file will be added to the input_file.json with the default values. Each answer from the user will update the json file.
* In case anything goes wrong, you can see the recovery files and logs using --verbose.
* You can use toml file format instead of json. Just use an `output_path` that ends with ".toml" instead of ".json".

## Examples
* To sort the elements by ELO score: `cat output_file.json | jq 'sort_by(.ELO)'`
