import git, os


def get_last_git_commit():
    """
    Get the last git revision in the current directory.
    Return:   string
    """
    repo = git.Repo(os.getcwd())
    return repo.commit('master').hexsha
