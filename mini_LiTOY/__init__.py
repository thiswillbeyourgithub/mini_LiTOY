import fire

from .mini_LiTOY import mini_LiTOY

def cli_launcher() -> None:
    fire.Fire(mini_LiTOY)

if __name__ == "__main__":
    cli_launcher()