
import nltk
from nltk.corpus import stopwords
from nltk.downloader import Downloader

def download_nltk_data():
    """Download necessary NLTK models. If models are already present, skip downloading."""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'wordnet': 'corpora/wordnet',
        'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger/english.pickle',
        'brown': 'corpora/brown',
        'omw-1.4': 'corpora/omw-1.4',
        'wordnet_ic': 'corpora/wordnet_ic',
        'punkt_tab': 'tokenizers/punkt/tab'
    }

    _downloader = Downloader()
    missing = [pkg for pkg in resources.keys() if not _downloader.is_installed(pkg)]
    if missing:
        print("Missing NLTK resources (will download):", missing)
        for pkg in missing:
            _downloader.download(pkg, quiet=True)
        print("Downloaded missing NLTK resources.")
    else:
        print("All required NLTK resources are already installed.")

    return set(stopwords.words('english'))