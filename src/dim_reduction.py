# imports
import logging

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


# functions
def do_assertion(data, n_dimensions):
    assert len(data) >= n_dimensions, 'You must have same or more data than' \
                                      ' n_dimensions.'


def truncated_svd(data, n_dimensions, n_iter=7, random_state=42):
    logging.info('getting truncated svd')
    do_assertion(data, n_dimensions)

    svd = TruncatedSVD(n_components=n_dimensions,
                       n_iter=n_iter,
                       random_state=random_state)

    reducted = svd.fit_transform(data)

    return reducted, svd


def pca(data, n_dimensions):
    logging.info('getting pca')
    do_assertion(data, n_dimensions)

    pca = PCA(n_components=n_dimensions)

    reducted = pca.fit_transform(data)

    return reducted, pca
