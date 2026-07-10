"""Utility functions for memory debugging."""


def load_and_print_pickle_file(
    path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl",
):
    """Load and print the contents of a serialized replay dataset pickle file.

    Prints the number of samples (``len(data[0])``), the first element of each
    data component, and then each component in full.  Intended for offline
    inspection and debugging of persisted replay buffer datasets.

    Args:
        path: Filesystem path to the pickle file.  Defaults to the original
            developer's working path; pass an explicit path in practice.
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
