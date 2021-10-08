from git import Repo


def get_commit_message(rev: str) -> str:
    return Repo(".").commit(rev).summary
