# mini_LiTOY.md
Minimalist approach to the [LiTOY task sorting algorithm](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You) based on [ELO scores](https://en.wikipedia.org/wiki/Elo_rating_system).

# Features
* Minimalist
* Statically typed via typeguard
* Made to be extensible
* Supports callbacks

# Questions

### What is LiTOY?
[LiTOY](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You) was a personal project. The idea is to dump all your TODOs in one place, then rank them using ELO scores based on how important they are and of how fast they are.

### Why make mini_LiTOY?
mini_LiTOY's idea is to keep the code idea but this time in a minimalist python script. As long as the user takes care of storing the tasks in a text files, the LiTOY algorithm will update an output json based on the score. The first use for this will be to rank my [Omnivore](https://github.com/omnivore-app/omnivore) reading queue by downloading titles of articles to read via their API, then uploading the ELO score as a label. This can be found in the examples folder.

# Usage
* `python -m pip install mini_LiTOY` or `python -m pip install git+https://github.com/thiswillbeyourgithub/mini_LiTOY.git`
* `python -m mini_LiTOY --input_file my_text_file.txt --output_json output_file.json`
* Note: each new (nonempty nor commented) line in input_file will be added to the input_file.json with the default values. Each answer from the user will update the json file.
* In case anything goes wrong, you can see the recovery files and logs using --verbose.

## Examples
* To sort the elements by ELO score: `cat output_file.json | jq 'sort_by(.ELO)'`
