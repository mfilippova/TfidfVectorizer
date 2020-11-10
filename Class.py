import math


class CountVectorizer():
    """Class CountVectorizer for encoding texts"""

    def __init__(self, lowercase=True):
        self.lowercase = lowercase
        self._vocabulary = []

    def get_feature_names(self, ):
        return self._vocabulary

    def fit_transform(self, corpus):
        if self.lowercase:
            splitted_corpus = [doc.lower().split() for doc in corpus]
        else:
            splitted_corpus = [doc.split() for doc in corpus]

        bag_of_words = set()
        for doc in splitted_corpus:
            for word in doc:
                if word not in bag_of_words:
                    bag_of_words.add(word)
                    self._vocabulary.append(word)

        vectors = [[doc.count(word) for word in self._vocabulary]
                   for doc in splitted_corpus]
        return vectors


class TfidfTransformer():
    """
    Class TfidfTransformer for transforming matrix into Tfidf form
    """
    def tf_transform(self, matrix):
        tf = []
        for x in matrix:
            tf_row = []
            sum_row = sum(x)
            for item in x:
                tf_row.append(item/sum_row)
            tf.append(tf_row)
        return tf

    def idf_transform(self, matrix):
        idf = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    idf[j] += 1
        len_matrix = len(matrix)
        for i in range(len(idf)):
            idf[i] = math.log((len_matrix + 1)/(idf[i] + 1)) + 1
        return idf

    def fit_transform(self, matrix):
        tf = self.tf_transform(matrix)
        idf = self.idf_transform(matrix)
        tfidf = []
        for x in tf:
            tfidf_row = []
            for i in range(len(x)):
                tfidf_row.append(x[i]*idf[i])
            tfidf.append(tfidf_row)
        return tfidf


class TfidfVectorizer(CountVectorizer, TfidfTransformer):
    """
    Class TfidfVectorizer transforms corpus into Tfidf matrix
    """
    def __init__(self, lowercase=True):
        super().__init__(lowercase)

    def fit_transform(self, corpus):
        matrix = CountVectorizer.fit_transform(self, corpus)
        tfidf = TfidfTransformer.fit_transform(self, matrix)
        return tfidf


if __name__ == "__main__":
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(tfidf_matrix)
