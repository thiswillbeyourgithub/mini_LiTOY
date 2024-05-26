import fire

class LiTOY:
    def __init__(self, input_file=None):
        self.lines = []
        if input_file:
            with open(input_file, 'r') as file:
                self.lines = file.readlines()

if __name__ == "__main__":
    fire.Fire(LiTOY)
