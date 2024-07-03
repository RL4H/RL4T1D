from environment.utils import risk_index


def time_in_range(CGM):
    '''
    slightly modified/latest way to calc the medical metrics
    :param CGM: an array with the cgm readings
    :return: time in ranges hypo, hyper, normo
    '''
    hypo, normo, hyper, severe_hypo, severe_hyper = 0, 0, 0, 0, 0

    if len(CGM) == 0:
        CGM.append(0)  # to avoid division by zero

    for reading in CGM:
        if reading <= 54:
            severe_hypo += 1
        elif reading <= 70:
            hypo += 1
        elif reading <= 180:
            normo += 1
        elif reading <= 250:
            hyper += 1
        else:
            severe_hyper += 1

    LBGI, HBGI, RI = risk_index(CGM, len(CGM))

    # todo add other useful metrics, capability to save metrics to a file.
    return (normo * 100 / len(CGM)), (hypo * 100 / len(CGM)), (severe_hypo * 100 / len(CGM)), \
           (hyper * 100 / len(CGM)), LBGI, HBGI, RI, (severe_hyper * 100 / len(CGM))
