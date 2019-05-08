from nltk.tokenize import word_tokenize
import string
from tqdm import tqdm
import glob

def concate():
    train_pos = glob.glob("train/pos/*.txt")
    train_neg = glob.glob("train/neg/*.txt")
    test_pos = glob.glob("test/pos/*.txt")
    test_neg = glob.glob("test/neg/*.txt")
    files_dict = {'train_pos':train_pos,'train_neg':train_neg,'test_pos':test_pos,'test_neg':test_neg}
    for name,files in files_dict.items():
        with open(name+".txt", "wb") as outfile:
            for f in files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read()+b'\n')

def clean_txt(filename):
    # read text
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    #lower case
    tokens = word_tokenize(text)
    tokens = [w.lower() for w in tokens]

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    output_txt = ' '.join(words)

    f2 = open(filename,'w+')
    f2.write(output_txt)
    f2.close()


def loop_files(file_list):
    for path in tqdm(file_list):
        files = glob.glob(path)
        for file in tqdm(files):
            clean_txt(file)


file_list = ["train/pos/*.txt","train/neg/*.txt","test/pos/*.txt","test/neg/*.txt"]
loop_files(file_list)
concate()