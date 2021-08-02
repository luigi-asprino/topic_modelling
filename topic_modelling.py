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
from abc import abstractmethod

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


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]


class CorpusInput(object):

    @abstractmethod
    def split(self, n_of_chunks): raise NotImplementedError


class CorpusInputFolder(CorpusInput):

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files_to_process = []
        logger.info(f"scanning {self.folder_path}")
        for (dirpath, dirname, filenames) in os.walk(self.folder_path):
            for f in filenames:
                self.files_to_process.append(os.path.join(dirpath, f))

    def split(self, n_of_chunks):
        chunks = split_list(self.files_to_process, n_of_chunks)
        result = []
        for chunk in chunks:
            result.append(CorpusInputFolderChunk(chunk))
        return result

class CorpusRocksDB(CorpusInput):

    def __init__(self, rocks_db_path):
        self.rocks_db_path = rocks_db_path
        logger.info(f"opening {self.rocks_db_path}")
        self.db = rocksdb.DB(self.rocks_db_path, rocksdb.Options(create_if_missing=True))

        self.docs_to_process = []
        it = self.db.iterkeys()
        it.seek_to_first()
        for k in list(it):
            self.docs_to_process.append(k.decode())

        logger.info(f"number of docs {len(self.docs_to_process)}")

    def split(self, n_of_chunks):
        chunks = split_list(self.docs_to_process, n_of_chunks)
        result = []
        for chunk in chunks:
            result.append(CorpusRocksDBChunk(chunk, self.db))
        return result


class CorpusChunk(object):

    @abstractmethod
    def get_documents(self): raise NotImplementedError


class CorpusInputFolderChunk(CorpusChunk):

    def __init__(self, files):
        self.files = files

    def __iter__(self):
        for file in self.files:
            yield DocumentInputFolder(file)

    def get_documents(self):
        result = []
        for file in self.files:
            result.append(DocumentInputFolder(file))
        return result

class CorpusRocksDBChunk(CorpusChunk):

    def __init__(self, ids, db):
        self.ids = ids
        self.db = db

    def __iter__(self):
        for id in self.ids:
            yield DocumentRocksDB(id, self.db)

class Document(object):

    @abstractmethod
    def lines(self): raise NotImplementedError

    @abstractmethod
    def id(self): raise NotImplementedError


class DocumentInputFolder(Document):

    def __init__(self, path):
        self.path = path

    def lines(self):
        return open(self.path, "r")

    def id(self):
        return self.path

class DocumentRocksDB(Document):

    def __init__(self, id, db):
        self.id_db = id
        self.db = db

    def lines(self):
        return [self.db.get(self.id_db.encode()).decode()]

    def id(self):
        return self.id_db


class CorpusMP(object):

    def __init__(self, chunk, dictionary, fw_progress, fw_index):
        self.chunk = chunk
        self.dictionary = dictionary
        self.fw_progress = fw_progress
        self.fw_index = fw_index

    def __iter__(self):
        count = 0
        for doc in self.chunk:
            if (count % 100 == 0):
                self.fw_progress.write(str(count) + "\n")
            counter, error = get_counter(doc, self.dictionary)
            self.fw_index.write(f"{doc.id()}\tf{error}\n")
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


def regTokenize(text):
    words = WORD.findall(text)
    return words


def add_language_tags(s):
    for c in pycountry.countries:
        s.add(c.alpha_2.lower())
        s.add(c.alpha_3.lower())


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


def filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, tokens_to_keep):
    logger.info(f"Filter dictionary")
    dictionary.filter_extremes(no_below=filter_no_below, no_above=filter_no_above, keep_n=None,
                               keep_tokens=tokens_to_keep)

    # logger.info(f"Dictionary length before stopword remove {len(dictionary)}")
    stopword_ids = [id for id, f in dictionary.doc2bow([str(s).lower() for s in stop])]
    dictionary.filter_tokens(bad_ids=stopword_ids)


def get_chunk_path(dictionary_base, number):
    return dictionary_base + "_" + str(number)


def get_dictionary(dictionary_file, corpus_input, stop, filter_no_below, filter_no_above, keepwords,
                   filter_every=1000, multiproces=False):
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

        if multiproces:
            chunks = corpus_input.split(os.cpu_count())
            processes = []
            for p_number in range(0, os.cpu_count()):
                process = Process(target=get_dictionary_chunk, args=(
                    get_chunk_path(dictionary_file, p_number), chunks[p_number], stop, filter_no_below,
                    filter_no_above, keepwords, p_number, filter_every))
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
            logger.info(f"Dictionary computed in {str(dictionaryTimeTotal)}, length {len(dictionary)}")
        else:
            logger.info(f"No multiprocess")
            chunks = corpus_input.split(1)
            get_dictionary_chunk(dictionary_unfiltered_path, chunks[0], stop, filter_no_below, filter_no_above, keepwords,
                                 0, filter_every=1000)
            dictionary = corpora.Dictionary.load(dictionary_unfiltered_path)
            filter_dictionary(dictionary, filter_no_below, filter_no_above, stop, keepwords)
            dictionary.save(dictionary_file)

        logger.info(f"Dictionary of length {len(dictionary)}")

    return dictionary


def get_counter(doc, dictionary=None):
    counter = {}
    error = False
    try:
        for line in doc.lines():
            tokenized_line = regTokenize(str(line).lower())
            if (dictionary is not None):
                tokenized_line = dictionary.doc2idx(tokenized_line)
            for w in tokenized_line:
                if w in counter:
                    counter[w] += 1
                else:
                    counter[w] = 1
    except Exception as e:
        logger.error(f"Errror while reading \"{doc.id()}\"")
        error = True
    if (dictionary is not None):
        return list(counter.items()), error
    return counter, error


def add_document_to_dictionary(doc, dict):
    dict.add_document_counter(get_counter(doc)[0])


def get_dictionary_chunk(dictionary_file, chunk, stop, filter_no_below, filter_no_above, keepwords,
                         process_number, filter_every=1000):
    dictionaryTime0 = time.time()
    logger.info(
        f"Process {process_number} - Computing dictionary chunk - filter every {filter_every}, dictionary file {dictionary_file}")
    dictionary = DictionaryCounter(prune_at=None)
    processed_documents = 0
    fw_progress = open(dictionary_file + "_progress", 'w')
    for doc in chunk:
        if (processed_documents > 0 and processed_documents % 100 == 0):
            dictionaryTime1 = time.time()
            logger.info(
                f"Process {process_number} - Processed {processed_documents} documents {(dictionaryTime1 - dictionaryTime0) / processed_documents}s per document within a dictionary chunk, length {len(dictionary)}")
            fw_progress.write(str(processed_documents) + "\n")
        add_document_to_dictionary(doc, dictionary)
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
        f"Process {process_number} - Dictionary chunk computed in {str(dictionaryTimeTotal)}, length {len(dictionary)}, file {dictionary_file}")


def process_chunk(chunk, dictionary, corpusMM_file):
    fw_progress = open(corpusMM_file + "_progress", 'w')
    fw_index = open(corpusMM_file + "_index", 'w')
    corpus_memory_friendly = CorpusMP(chunk, dictionary, fw_progress, fw_index)
    corpora.MmCorpus.serialize(corpusMM_file, corpus_memory_friendly)
    logger.info(f"{corpusMM_file} written!")
    fw_progress.flush()
    fw_progress.close()
    fw_index.flush()
    fw_index.close()


def get_tfidf_corpus(tfidf_corpus_file, corpus_input, dictionary, corpusMM_file, tfidf_model_file, multiprocess = True):

    logger.info(f"Using dictionary of length {len(dictionary)}")

    # if corpus not exists
    if not os.path.isfile(tfidf_corpus_file):
        # compute corpus
        corpusTime0 = time.time()
        logger.info("Computing corpus multi-process")

        if multiprocess:

            chunks = corpus_input.split(os.cpu_count())

            processes = []
            for p_number in range(0, os.cpu_count()):
                process = Process(target=process_chunk, args=(
                    chunks[p_number], dictionary, get_chunk_path(corpusMM_file, p_number)))
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

        else:
            logger.info("Computing corpus no multiprocess")
            chunks = corpus_input.split(1)
            process_chunk(chunks[0], dictionary, corpusMM_file)

        corpus_memory_friendly = corpora.MmCorpus(corpusMM_file)
        corpusTime1 = time.time()
        corpusTimeTotal = corpusTime1 - corpusTime0
        logger.info(
            f"Corpus computed {len(corpus_memory_friendly)} and loaded from {corpusMM_file} in {str(corpusTimeTotal)}")
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

def printTopics(outdir, lda, corpus_input, corpus_lda):
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

    # fw_doc_id_to_path = open(doc_id_to_path_filepath, 'w')
    # counter = 0
    #
    # if os.path.isfile(corpus_dir):
    #     with open(corpus_dir, "r") as fp:
    #         line = fp.readline()
    #         if (len(line[:-1]) > 1):
    #             fw_doc_id_to_path.write("Doc #")
    #             fw_doc_id_to_path.write(str(counter))
    #             fw_doc_id_to_path.write("\t")
    #             fw_doc_id_to_path.write(line[:-1])
    #             fw_doc_id_to_path.write("\n")
    #             counter += 1
    #
    # elif os.path.isdir(corpus_dir):
    #     for (dirpath, dirname, filenames) in os.walk(corpus_dir):
    #         for f in filenames:
    #             fw_doc_id_to_path.write("Doc #")
    #             fw_doc_id_to_path.write(str(counter))
    #             fw_doc_id_to_path.write("\t")
    #             fw_doc_id_to_path.write(os.path.join(dirpath, f))
    #             fw_doc_id_to_path.write("\n")
    #             counter += 1
    #
    # fw_doc_id_to_path.close()


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

def compute_topics(outdir, corpus_input, filter_no_below, filter_no_above,
                   num_of_topics=[2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16], stopwords_file="stopwords.txt",
                   keepwords_file="keepwords.txt", multiprocess=True):

    if (not os.path.exists(outdir)):
        os.mkdir(outdir)

    dictionary_file = outdir + "/dictionary"
    tfidf_model_file = outdir + "/tfidf_model"
    tfidf_corpus_file = outdir + "/tfidf_corpus"
    corpusMM_file = outdir + "/corpus"

    # get stopwords
    stop = get_stopwords(stopwords_file)

    # loads tokens to keep
    keepwords = get_keepwords(keepwords_file)

    # Create the dictionary
    dictionary = get_dictionary(dictionary_file, corpus_input, stop, filter_no_below, filter_no_above, keepwords,
                                1000, multiprocess)

    # Get corpus TF-IDF
    corpus, corpus_tfidf = get_tfidf_corpus(tfidf_corpus_file, corpus_input, dictionary, corpusMM_file,
                                            tfidf_model_file, multiprocess)

    if multiprocess:
        processes = []

        # # TF-IDF
        for n_of_topics in num_of_topics:
            process = Process(target=computeAndPrintTopics,
                              args=(outdir, n_of_topics, corpus_tfidf, dictionary, corpus_input, "TF-IDF"))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    else:
        logger.info("Computing topics")
        for n_of_topics in num_of_topics:
            logger.info(f"Compute {n_of_topics} topics")
            computeAndPrintTopics(outdir, n_of_topics, corpus_tfidf, dictionary, corpus_input, "TF-IDF")

def compute_similarity(outdir, corpus_input, filter_no_below, filter_no_above, stopwords_file="stopwords.txt",
                   keepwords_file="keepwords.txt", multiprocess=True, n_of_topics=500):

    if (not os.path.exists(outdir)):
        os.mkdir(outdir)

    dictionary_file = outdir + "/dictionary"
    tfidf_model_file = outdir + "/tfidf_model"
    tfidf_corpus_file = outdir + "/tfidf_corpus"
    corpusMM_file = outdir + "/corpus"
    output_prefix = outdir + "/output_prefix"

    # get stopwords
    stop = get_stopwords(stopwords_file)

    # loads tokens to keep
    keepwords = get_keepwords(keepwords_file)

    # Create the dictionary
    dictionary = get_dictionary(dictionary_file, corpus_input, stop, filter_no_below, filter_no_above, keepwords,
                                1000, multiprocess)

    # Get corpus TF-IDF
    corpus, corpus_tfidf = get_tfidf_corpus(tfidf_corpus_file, corpus_input, dictionary, corpusMM_file,
                                            tfidf_model_file, multiprocess)

    logger.info("Computing LSI model")

    lsi = models.LsiModel(corpus_tfidf, num_topics=n_of_topics)

    from gensim.similarities import Similarity

    logger.info("Computing similarity among the documents of the corpus")

    index = Similarity(output_prefix, lsi, num_features=n_of_topics)

    for similarities in index:
        print(similarities)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='...')

    parser.add_argument('--corpus_folder', dest='corpus_folder',
                        help='Path to the folder containing the corpus.')
    parser.add_argument('--corpus_rocks_db', dest='corpus_rocks_db',
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
    parser.add_argument('--clean', dest='clean', help='Delete the output directory before computing the model',
                        action='store_true')
    parser.add_argument('--similarities', dest='similarities', help='Compute the similarities among the documents of the corpus',
                        action='store_true')

    args = parser.parse_args()

    logger.info("Topic model extractor")

    t0 = time.time()

    if args.clean and os.path.exists(args.output_directory):
        import shutil
        shutil.rmtree(args.output_directory)

    if args.corpus_folder is not None:
        corpus = CorpusInputFolder(args.corpus_folder)
        multiprocess = True
    elif args.corpus_rocks_db is not None:
        corpus = CorpusRocksDB(args.corpus_rocks_db)
        multiprocess = False

    if args.similarities is None:
        compute_topics(args.output_directory, corpus, int(args.no_below), float(args.no_above),
                   args.topics.split(","),
                   args.stopwords, args.keep_words, multiprocess)
    else:
        compute_similarity(args.output_directory, corpus, int(args.no_below), float(args.no_above),
                       args.stopwords, args.keep_words, multiprocess)

    t1 = time.time()
    total = t1 - t0
    timings.info("Total time " + str(total))
