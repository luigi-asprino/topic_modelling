import bz2
from english_words import english_words_lower_alpha_set
from gensim import corpora, models
from itertools import chain
import logging
from multiprocessing import Process
import os
import pycountry
import re
from stop_words import get_stop_words, AVAILABLE_LANGUAGES
import time
import rocksdb

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("polyglot").setLevel(logging.ERROR)
trace = logging.getLogger("trace")
trace.setLevel(logging.INFO)
timings = logging.getLogger("timings")
timings.setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

DOC_LEN_SPLIT = 100000000

FILTER_CHUNKS = False

WORD = re.compile(r'[a-zA-Z][a-zA-Z]+')

def regTokenize(text):
    words = WORD.findall(text)
    return words

class Corpus(object):

    def __init__(self, corpus_folder, dictionary, stopwords):
        self.corpus_folder = corpus_folder
        self.stopwords = stopwords
        self.dictionary = dictionary
        self.counter = 0

    def __iter__(self):

        for root, dirs, files in os.walk(self.corpus_folder):
            for filename in files:
                if (filename == "virtualdocument.txt.bz2"):
                    self.counter = self.counter + 1
                    if (self.counter % 1000 == 0):
                        logger.info(f"Processed {self.counter} documents ")
                    for d in DocumentSplitter(os.path.join(root, filename)):
                        yield self.dictionary.doc2bow(d)


class CorpusMP(object):

    def __init__(self, files, dictionary, fw_progress, fw_index):
        self.files = files
        self.dictionary = dictionary
        self.fw_progress = fw_progress
        self.fw_index = fw_index

    def __iter__(self):
        count = 0
        for file in self.files:
            if (count % 100 == 0):
                self.fw_progress.write(str(count) + "\n")
            counter, error = get_counter(file, self.dictionary)
            self.fw_index.write(f"{file}\tf{error}\n")
            count += 1
            yield counter


class DictionaryCounter(corpora.Dictionary):

    def add_document_counter(self, counter):

        token2id = self.token2id
        missing = sorted(x for x in counter.items() if x[0] not in token2id)
        for w, _ in missing:
            token2id[w] = len(token2id)
        result = {token2id[w]: freq for w, freq in counter.items() if w in token2id}

        self.num_docs += 1
        self.num_pos += sum(counter.values())
        self.num_nnz += len(result)
        for tokenid, freq in result.items():
            self.cfs[tokenid] = self.cfs.get(tokenid, 0) + freq
            self.dfs[tokenid] = self.dfs.get(tokenid, 0) + 1


class DocumentSplitter(object):

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        l = 0
        doc = []
        try:
            for line in bz2.open(self.filename, "r"):
                if (l > DOC_LEN_SPLIT):
                    l = 0
                    result = doc.copy()
                    doc = []
                    yield result
                else:
                    l = l + len(line)
                    doc.extend(regTokenize(str(line).lower()))
        except:
            logger.error(f"Error while reading {self.filename}")
            yield doc
        if len(doc) > 0:
            yield doc


def get_counter(filename, dictionary=None):
    counter = {}
    error = False
    try:
        if filename.endswith("bz2"):
            file = bz2.open(filename, "r")
        elif filename.endswith("txt"):
            file = open(filename,"r")

        for line in file:
            tokenized_line = regTokenize(str(line).lower())
            if (dictionary is not None):
                tokenized_line = dictionary.doc2idx(tokenized_line)
            for w in tokenized_line:
                if w in counter:
                    counter[w] += 1
                else:
                    counter[w] = 1
    except Exception as e:
        logger.error(f"Errror while reading \"{filename}\"")
        error = True
    if (dictionary is not None):
        return list(counter.items()), error
    return counter, error

def get_counter_db(filename, db, dictionary=None):
    counter = {}
    error = False
    try:

        tokenized_line = regTokenize(str(db.get(filename.encode()).decode()).lower())
        if (dictionary is not None):
            tokenized_line = dictionary.doc2idx(tokenized_line)

        for w in tokenized_line:
            if w in counter:
                counter[w] += 1
            else:
                counter[w] = 1

    except Exception as e:
        logger.error(f"Errror while reading \"{filename}\"")
        error = True
    if (dictionary is not None):
        return list(counter.items()), error
    return counter, error


def add_document_to_dictionary(filename, dict):
    dict.add_document_counter(get_counter(filename)[0])

def add_document_to_dictionary_db(filename, dict, db):
    dict.add_document_counter(get_counter_db(filename)[0])


def add_language_tags(s):
    for c in pycountry.countries:
        s.add(c.alpha_2.lower())
        s.add(c.alpha_3.lower())


def filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, tokens_to_keep):
    logger.info(f"Filter dictionary")
    dictionary.filter_extremes(no_below=filter_no_below, no_above=filter_no_above, keep_n=None,
                               keep_tokens=tokens_to_keep)

    # logger.info(f"Dictionary length before stopword remove {len(dictionary)}")
    stopword_ids = [id for id, f in dictionary.doc2bow([str(s).lower() for s in stop])]
    dictionary.filter_tokens(bad_ids=stopword_ids)


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


def split_corpus(filepath):
    files_to_process = []
    if os.path.isfile(filepath):
        with open(filepath, "r") as fp:
            line = fp.readline()
            if (len(line[:-1]) > 0):
                files_to_process.append(line[:-1])
            while line:
                line = fp.readline()
                if (len(line[:-1]) > 0):
                    files_to_process.append(line[:-1])
    elif os.path.isdir(filepath):
        for (dirpath, dirname, filenames) in os.walk(filepath):
            for f in filenames:
                files_to_process.append(os.path.join(dirpath, f))

    return split_list(files_to_process, os.cpu_count()), files_to_process


def get_doc_ids(db):
    docs_to_process = []
    it = db.iterkeys()
    for k in list(it):
        docs_to_process.append(k.decode())
    return split_list(docs_to_process, os.cpu_count()), docs_to_process


def get_dictionary_chunk(dictionary_file, files_to_process, stop, filter_no_below, filter_no_above, keepwords,
                         process_number, filter_every=1000):
    dictionaryTime0 = time.time()
    logger.info(
        f"Process {process_number} - Computing dictionary chunk - filter every {filter_every}, dictionary file {dictionary_file}, files to process {len(files_to_process)}")
    dictionary = DictionaryCounter(prune_at=None)
    processed_documents = 0
    fw_progress = open(dictionary_file + "_progress", 'w')
    for filename in files_to_process:
        if (processed_documents > 0 and processed_documents % 100 == 0):
            dictionaryTime1 = time.time()
            logger.info(
                f"Process {process_number} - Processed {processed_documents} documents {(dictionaryTime1 - dictionaryTime0) / processed_documents}s per document within a dictionary chunk, length {len(dictionary)}")
            fw_progress.write(str(processed_documents) + "\n")
        add_document_to_dictionary(filename, dictionary)
        processed_documents += 1
        if (processed_documents % filter_every == 0):
            if os.path.isfile(dictionary_file):
                dictionary_load = DictionaryCounter.load(dictionary_file)
                dictionary.merge_with(dictionary_load)
            if (FILTER_CHUNKS):
                filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
            dictionary.save(dictionary_file)
            dictionary = DictionaryCounter(prune_at=None)
    if os.path.isfile(dictionary_file):
        dictionary_load = DictionaryCounter.load(dictionary_file)
        dictionary.merge_with(dictionary_load)
    if (FILTER_CHUNKS):
        filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
    dictionary.save(dictionary_file)
    fw_progress.write(str(processed_documents) + "\n")
    fw_progress.close()
    dictionaryTime1 = time.time()
    dictionaryTimeTotal = dictionaryTime1 - dictionaryTime0
    logger.info(
        f"Process {process_number} - Dictionary chunk computed in {str(dictionaryTimeTotal)}, length {len(dictionary)}, {dictionaryTimeTotal / len(files_to_process)}s per document, file {dictionary_file}")



def get_dictionary_chunk_db(dictionary_file, files_to_process, stop, filter_no_below, filter_no_above, keepwords,
                         process_number, db, filter_every=1000):
    dictionaryTime0 = time.time()
    logger.info(
        f"Process {process_number} - Computing dictionary chunk - filter every {filter_every}, dictionary file {dictionary_file}, files to process {len(files_to_process)}")
    dictionary = DictionaryCounter(prune_at=None)
    processed_documents = 0
    fw_progress = open(dictionary_file + "_progress", 'w')
    for filename in files_to_process:
        if (processed_documents > 0 and processed_documents % 100 == 0):
            dictionaryTime1 = time.time()
            logger.info(
                f"Process {process_number} - Processed {processed_documents} documents {(dictionaryTime1 - dictionaryTime0) / processed_documents}s per document within a dictionary chunk, length {len(dictionary)}")
            fw_progress.write(str(processed_documents) + "\n")
        add_document_to_dictionary_db(filename, dictionary,db)
        processed_documents += 1
        if (processed_documents % filter_every == 0):
            if os.path.isfile(dictionary_file):
                dictionary_load = DictionaryCounter.load(dictionary_file)
                dictionary.merge_with(dictionary_load)
            if (FILTER_CHUNKS):
                filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
            dictionary.save(dictionary_file)
            dictionary = DictionaryCounter(prune_at=None)
    if os.path.isfile(dictionary_file):
        dictionary_load = DictionaryCounter.load(dictionary_file)
        dictionary.merge_with(dictionary_load)
    if (FILTER_CHUNKS):
        filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
    dictionary.save(dictionary_file)
    fw_progress.write(str(processed_documents) + "\n")
    fw_progress.close()
    dictionaryTime1 = time.time()
    dictionaryTimeTotal = dictionaryTime1 - dictionaryTime0
    logger.info(
        f"Process {process_number} - Dictionary chunk computed in {str(dictionaryTimeTotal)}, length {len(dictionary)}, {dictionaryTimeTotal / len(files_to_process)}s per document, file {dictionary_file}")

def get_chunk_path(dictionary_base, number):
    return dictionary_base + "_" + str(number)


def get_dictionary_mp(dictionary_file, corpus_dir, stop, filter_no_below, filter_no_above, keepwords,
                      filter_every=1000, db=None):
    dictionaryTime0 = time.time()
    logger.info("Computing dictionary")
    dictionary = corpora.Dictionary(prune_at=None)
    dictionary_unfiltered_path = dictionary_file + "_unfiltered"
    if os.path.isfile(dictionary_unfiltered_path):
        logger.info("Loading dictionary")
        dictionary = corpora.Dictionary.load(dictionary_unfiltered_path)
        logger.info(f"Dictionary unfiltered loaded! Dictionary length {len(dictionary)}")
        filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
        dictionary.save(dictionary_file)
        logger.info(f"Dictionary loaded! Dictionary length {len(dictionary)}")
    else:

        if db is None:
            files_to_process_chunks, files_to_process = split_corpus(corpus_dir)
        else:
            files_to_process_chunks, files_to_process = get_doc_ids(db)

        processes = []
        for p_number in range(0, os.cpu_count()):
            process = Process(target=get_dictionary_chunk_db, args=(
                get_chunk_path(dictionary_file, p_number), files_to_process_chunks[p_number], stop, filter_no_below,
                filter_no_above, keepwords, p_number, db, filter_every))
            processes.append(process)
            process.start()
            logger.info(f"Process {p_number} started")

        for process in processes:
            process.join()

        logger.info("Merging dictionary chunks")
        for p_number in range(0, os.cpu_count()):
            if os.path.isfile(get_chunk_path(dictionary_file, p_number)):
                dictionary_load = corpora.Dictionary.load(get_chunk_path(dictionary_file, p_number))
                dictionary.merge_with(dictionary_load)

        logger.info(f"Length unfiltered dictionary {len(dictionary)}")
        dictionary.save(dictionary_unfiltered_path)
        filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
        dictionary.save(dictionary_file)
        dictionaryTime1 = time.time()
        dictionaryTimeTotal = dictionaryTime1 - dictionaryTime0
        logger.info(
            f"Dictionary computed in {str(dictionaryTimeTotal)}, length {len(dictionary)}, {dictionaryTimeTotal / len(files_to_process)}s per document")

    return dictionary


def get_stopwords(stopwords_file):
    # Create a stopword list
    stop = set()
    with open(stopwords_file, "r") as fp:
        line = fp.readline()
        stop.add(line[:-1])
        while line:
            line = fp.readline()
            stop.add(line[:-1])

    add_language_tags(stop)

    # # Importing stopwords for available languages https://github.com/Alir3z4/python-stop-words
    for l in AVAILABLE_LANGUAGES:
        for sw in get_stop_words(l):
            stop.add(sw)

    logger.info(f"Number of Stopwords {len(stop)}")
    return stop


def get_keepwords(keepwords_file):
    # Create a stopword list
    tokens_to_keep = english_words_lower_alpha_set
    with open(keepwords_file, "r") as fp:
        line = fp.readline()[:-1]
        tokens_to_keep.add(line)
        while line:
            line = fp.readline()[:-1]
            tokens_to_keep.add(line)

    logger.info(f"Tokens to keep {len(tokens_to_keep)}")
    return tokens_to_keep


def process_chunk(files_to_process, dictionary, corpusMM_file):
    fw_progress = open(corpusMM_file + "_progress", 'w')
    fw_index = open(corpusMM_file + "_index", 'w')
    corpus_memory_friendly = CorpusMP(files_to_process, dictionary, fw_progress, fw_index)
    corpora.MmCorpus.serialize(corpusMM_file, corpus_memory_friendly)
    fw_progress.write(str(len(files_to_process)) + "\n")
    fw_progress.flush()
    fw_progress.close()
    fw_index.flush()
    fw_index.close()


def get_tfidf_corpus_mp(tfidf_corpus_file, corpus_dir, dictionary, stop, corpusMM_file, tfidf_model_file):
    # if corpus not exists
    if not os.path.isfile(tfidf_corpus_file):
        # compute corpus
        corpusTime0 = time.time()
        logger.info("Computing corpus multi-process")

        files_to_process_chunks, files_to_process = split_corpus(corpus_dir)
        processes = []
        for p_number in range(0, os.cpu_count()):
            process = Process(target=process_chunk, args=(
                files_to_process_chunks[p_number], dictionary, get_chunk_path(corpusMM_file, p_number)))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        corpus = []

        corpusMM_file_index = corpusMM_file + "_index"
        fw_index = open(corpusMM_file_index, "w")

        for p_number in range(0, os.cpu_count()):
            corpus_chunk_file = get_chunk_path(corpusMM_file, p_number)

            if os.path.isfile(corpus_chunk_file):
                corpus_chunk_file_index = corpus_chunk_file + "_index"
                chunk_index_reader = open(corpus_chunk_file_index, "r")
                fw_index.write(chunk_index_reader.read())

                loaded_corpus = corpora.MmCorpus(corpus_chunk_file)
                # logger.info(f"Documents within the loaded corpus {len(loaded_corpus)}")
                corpus = chain(corpus, loaded_corpus)
                # corpus.extend(loaded_corpus)

        fw_index.flush()
        fw_index.close()

        corpora.MmCorpus.serialize(corpusMM_file, corpus)
        corpus_memory_friendly = corpora.MmCorpus(corpusMM_file)
        corpusTime1 = time.time()
        corpusTimeTotal = corpusTime1 - corpusTime0
        logger.info(
            f"Corpus computed and serialized in {str(corpusTimeTotal)}, {corpusTimeTotal / len(files_to_process)}s per document")

        tfidfTime0 = time.time()
        logger.info("Computing TF-IDF model")
        tfidf = models.TfidfModel(corpus_memory_friendly, normalize=True)
        corpus_tfidf = tfidf[corpus_memory_friendly]
        corpora.MmCorpus.serialize(tfidf_corpus_file, corpus_tfidf)
        tfidf.save(tfidf_model_file)
        tfidfTime1 = time.time()
        tfidfTimeTotal = tfidfTime1 - tfidfTime0
        logger.info(f"TF-IDF model computed in {str(tfidfTimeTotal)}")

    else:
        # load corpus
        logger.info("Loading MM TF-IDF corpus")
        corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
        corpus_memory_friendly = corpora.MmCorpus(corpusMM_file)
        logger.info("TF IDF Corpus loaded")

    return corpus_memory_friendly, corpus_tfidf


def get_tfidf_corpus(tfidf_corpus_file, corpus_dir, dictionary, stop, corpusMM_file, tfidf_model_file):
    # if corpus not exists
    if not os.path.isfile(tfidf_corpus_file):
        # compute corpus
        corpusTime0 = time.time()
        logger.info("Computing corpus single process")
        corpus_memory_friendly = Corpus(corpus_dir, dictionary, stop)
        corpora.MmCorpus.serialize(corpusMM_file, corpus_memory_friendly)
        corpus_memory_friendly = corpora.MmCorpus(corpusMM_file)
        corpusTime1 = time.time()
        corpusTimeTotal = corpusTime1 - corpusTime0
        logger.info(
            f"Corpus computed and serialized in {str(corpusTimeTotal)}, {corpusTimeTotal / len(corpus_memory_friendly)}s per document")

        tfidfTime0 = time.time()
        logger.info("Computing TF-IDF model")
        tfidf = models.TfidfModel(corpus_memory_friendly, normalize=True)
        corpus_tfidf = tfidf[corpus_memory_friendly]
        corpora.MmCorpus.serialize(tfidf_corpus_file, corpus_tfidf)
        tfidf.save(tfidf_model_file)
        tfidfTime1 = time.time()
        tfidfTimeTotal = tfidfTime1 - tfidfTime0
        logger.info(f"TF-IDF model computed in {str(tfidfTimeTotal)}")

    else:
        # load corpus
        logger.info("Loading MM TF-IDF corpus")
        corpus_memory_friendly = corpora.MmCorpus(corpusMM_file)
        corpus_tfidf = corpora.MmCorpus(tfidf_corpus_file)
        logger.info("TF IDF Corpus loaded")

    return corpus_memory_friendly, corpus_tfidf


def compute_lda_model(outdir, n_of_topics, corpus_tfidf, dictionary, name):
    n_of_topics_folder = outdir + "/" + str(n_of_topics) + "_" + name
    lda_corpus_file = n_of_topics_folder + "/lsi_corpus"
    lda_model_file = n_of_topics_folder + "/lsi_model"

    if (not os.path.exists(n_of_topics_folder)):
        os.mkdir(n_of_topics_folder)

    lda_time_0 = time.time()
    logger.info(f"Computing LDA model for {n_of_topics} topics - {name}")
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=n_of_topics)
    corpus_lda = lda[corpus_tfidf]
    corpora.MmCorpus.serialize(lda_corpus_file, corpus_lda)
    lda.save(lda_model_file)
    logger.info(f"Perplexity for n of topics {n_of_topics}: {lda.log_perplexity(corpus_lda)} - {name}")
    lda_time_1 = time.time()
    lda_time_total = lda_time_1 - lda_time_0
    logger.info(f"LDA model  for {n_of_topics} topics computed in {str(lda_time_total)} - {name}")
    return lda, corpus_lda


def printTopics(outdir, lda, corpus_dir, corpus_lda):
    topics_filepath = outdir + "/topics.txt"
    doc_id_to_topic_filepath = outdir + "/doc_id_to_topic.txt"
    doc_id_to_path_filepath = outdir + "/doc_id_to_path.txt"
    writing0 = time.time()
    # logger.info("Saving topics")
    fw_topics = open(topics_filepath, 'w')
    for tid, topic in lda.show_topics(formatted=False, num_words=100):
        fw_topics.write("Topic #")
        fw_topics.write(str(tid))
        fw_topics.write("\n")
        for word, weight in topic:
            fw_topics.write("\t")
            fw_topics.write(word)
            fw_topics.write("\t")
            fw_topics.write(str(weight))
            fw_topics.write("\n")

    fw_doc_id_to_path = open(doc_id_to_path_filepath, 'w')
    counter = 0

    if os.path.isfile(corpus_dir):
        with open(corpus_dir, "r") as fp:
            line = fp.readline()
            if (len(line[:-1]) > 1):
                fw_doc_id_to_path.write("Doc #")
                fw_doc_id_to_path.write(str(counter))
                fw_doc_id_to_path.write("\t")
                fw_doc_id_to_path.write(line[:-1])
                fw_doc_id_to_path.write("\n")
                counter += 1

    elif os.path.isdir(corpus_dir):
        for (dirpath, dirname, filenames) in os.walk(corpus_dir):
            for f in filenames:
                fw_doc_id_to_path.write("Doc #")
                fw_doc_id_to_path.write(str(counter))
                fw_doc_id_to_path.write("\t")
                fw_doc_id_to_path.write(os.path.join(dirpath, f))
                fw_doc_id_to_path.write("\n")
                counter += 1

    fw_doc_id_to_path.close()


    # with open(corpus_dir, "r") as fp:
    #     line = fp.readline()
    #     if (len(line[:-1]) > 1):
    #         fw_doc_id_to_path.write("Doc #")
    #         fw_doc_id_to_path.write(str(counter))
    #         fw_doc_id_to_path.write("\t")
    #         fw_doc_id_to_path.write(line[:-1])
    #         fw_doc_id_to_path.write("\n")
    #         counter += 1
    # fw_doc_id_to_path.close()

    # logger.info("Saving DocId to topics")
    fw_docid_to_topics = open(doc_id_to_topic_filepath, 'w')
    for idx, doc in enumerate(corpus_lda):
        ss = sorted(doc, key=lambda i: i[1])
        fw_docid_to_topics.write("Doc #")
        fw_docid_to_topics.write(str(idx))
        fw_docid_to_topics.write("\n")
        for topic_id, w in ss:
            fw_docid_to_topics.write("\t")
            fw_docid_to_topics.write(str(w))
            fw_docid_to_topics.write(" on topic #")
            fw_docid_to_topics.write(str(topic_id))
            fw_docid_to_topics.write("\n")

    fw_docid_to_topics.close()
    writing1 = time.time()
    writingTimeTotal = writing1 - writing0
    # timings.info("Writing Time " + str(writingTimeTotal)) 


def computeAndPrintTopics(outdir, n_of_topics, corpus_tfidf, dictionary, corpus_dir, name):
    lda, corpus_lda = compute_lda_model(outdir, n_of_topics, corpus_tfidf, dictionary, name)
    printTopics(outdir + "/" + str(n_of_topics) + "_" + name, lda, corpus_dir, corpus_lda)


def compute_topics(outdir, corpus_dir, filter_no_below, filter_no_above,
                   num_of_topics=[2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16], stopwords_file="stopwords.txt",
                   keepwords_file="keepwords.txt", is_rocks_db_corpus=False):
    if (not os.path.exists(outdir)):
        os.mkdir(outdir)

    if is_rocks_db_corpus:
        db = rocksdb.DB(corpus_dir, rocksdb.Options(create_if_missing=True))

    dictionary_file = outdir + "/dictionary"
    tfidf_model_file = outdir + "/tfidf_model"
    tfidf_corpus_file = outdir + "/tfidf_corpus"
    corpusMM_file = outdir + "/corpus"

    # get stopwords
    stop = get_stopwords(stopwords_file)

    # loads tokens to keep 
    keepwords = get_keepwords(keepwords_file)

    # Create the dictionary
    dictionary = get_dictionary_mp(dictionary_file, corpus_dir, stop, filter_no_below, filter_no_above, keepwords,
                                   1000)

    # Get corpus TF-IDF
    corpus, corpus_tfidf = get_tfidf_corpus_mp(tfidf_corpus_file, corpus_dir, dictionary, stop, corpusMM_file,
                                               tfidf_model_file)

    processes = []

    # # TF-IDF 
    for n_of_topics in num_of_topics:
        process = Process(target=computeAndPrintTopics,
                          args=(outdir, n_of_topics, corpus_tfidf, dictionary, corpus_dir, "TF-IDF"))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='...')

    parser.add_argument('--corpus_input', dest='corpus_input',
                        help='Path to the list of files to process or to the folder containing them.')
    parser.add_argument('--corpus_input_rocks_db', dest='rocks_db_path',
                        help='Path to the rocks db containing documents to process.')
    parser.add_argument('--output_directory', dest='output_directory', help='Output directory.')
    parser.add_argument('--no_below', dest='no_below', default="1",
                        help='Discard tokens that appear in less than X documents.')
    parser.add_argument('--no_above', dest='no_above', default="1.0",
                        help='Discard tokens that appear in more than X documents.')
    parser.add_argument('--stopwords', dest='stopwords', default="stopwords.txt",
                        help='Filepath to the file containing a list of words to discard (one per line).')
    parser.add_argument('--keep_words', dest='keep_words', default="keepwords.txt",
                        help='Filepath to the file containing a list of words to keep (one per line).')
    parser.add_argument('--topics', dest='topics', default="1,2,4,8",
                        help='A comma separated list of integers indicating the number of topics to extract.')

    args = parser.parse_args()

    logger.info("Topic model extractor")

    t0 = time.time()

    if args.rocks_db_path is None:
        compute_topics(args.output_directory, args.corpus_input, int(args.no_below), float(args.no_above),
                   args.topics.split(","),
                   args.stopwords, args.keep_words, False)
    else:
        compute_topics(args.output_directory, args.rocks_db_path, int(args.no_below), float(args.no_above),
                       args.topics.split(","),
                       args.stopwords, args.keep_words, True)
    t1 = time.time()
    total = t1 - t0
    timings.info("Total time " + str(total))
