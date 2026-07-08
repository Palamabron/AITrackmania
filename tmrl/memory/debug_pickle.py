"""Utility functions for memory debugging."""


def load_and_print_pickle_file(
    path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl",
):
    """
    Loads and prints content from a pickle file specified by the path.
    Reads the pickle file and displays the number of samples along with their content.
    """
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")
    print("full data:")
    for i, d in enumerate(data):
        print(f"[{i}]: {d}")


if __name__ == "__main__":
    load_and_print_pickle_file()
