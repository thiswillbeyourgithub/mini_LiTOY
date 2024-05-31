# mini_LiTOY.md
Minimalist approach to the [LiTOY task sorting algorithm](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You)

# Questions
### What is LiTOY?
[LiTOY](https://github.com/thiswillbeyourgithub/LiTOY-aka-List-that-Outlives-You) was a personal project. The idea is to dump all your TODOs in one place, then rank them using ELO scores based on how important they are and of how fast they are.
### Why make mini_LiTOY?
mini_LiTOY's idea is to keep the code idea but this time in a minimalistic python script. As long as the user takes care of storing the tasks in a text files, the LiTOY algorithm will update an output json based on the score. The first use for this will be to rank my [Omnivore](https://github.com/omnivore-app/omnivore) reading queue by downloading titles of articles to read via their API, then uploading the ELO score as a label. This can be found in the examples folder.

# Usage
* `python -m pip install -r requirements.txt`
* `python mini_LiTOY.py --input_file my_text_file.txt --output_json output_file.json --question 'What's the relative importance of those items to you?'`
Each new (nonempty nor commented) line in input_file will be added to the input_file.json with the default values. Each answer from the user will update the json file.

To sort the elements by ELO score: `cat output_file.json | jq 'sort_by(.ELO)'`
