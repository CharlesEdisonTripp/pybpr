import numpy


def log_score(p, *args, **kwargs):
    return numpy.log(p)


def brier_score(p, *args, **kwargs):
    return (1 - p) ** 2 + (0 - (1 - p)) ** 2


def prob_score(p, *args, **kwargs):
    return p


def accuracy_score(p, *args, **kwargs):
    return 1 if p > 0.5 else 0.5 if p == 0.5 else 0


def compute_dcg(seq):
    # for position, (predicted, actual) in enumerate(seq):
    #     print(f'{position}, {predicted}, {actual} : {position + 2 }, {numpy.log(position + 2)}, {1.0 / numpy.log(position + 2)}')
    return sum(
        (
            actual / numpy.log(position + 2)
            for position, (predicted, actual) in enumerate(seq)
        )
    )


def compute_ndcg(seq):
    dcg = compute_dcg(sorted(seq, reverse=True))
    idcg = compute_dcg(
        sorted(((actual, actual) for predicted, actual in seq), reverse=True)
    )
    if idcg <= 1e-6:
        return 1.0
    #   print(f'dcg: {dcg} / idcg: {idcg}  ; {len(seq)}')
    # print(f'dcg: {dcg} / idcg: {idcg} = ndcg: {dcg / idcg} ; {len(seq)}')
    return dcg / idcg


# def compute_ndcg_binary(predicted, actual):
#     '''
#     + actual is 1 or 0
#     + dcg is sum(1/numpy.log(pos+1)) for all true positives
#     + idcg is sum(1/numpy.log(pos+1)) for pos = 1 .. # positive ratings
#     '''
#     dcg = compute_dcg(
#         sorted((
#             predicted
#             for predicted, actual in seq),
#         reverse=True))
#     idcg = compute_dcg(
#         sorted((
#             actual
#             for predicted, actual in seq),
#         reverse=True))
#     return dcg / idcg
