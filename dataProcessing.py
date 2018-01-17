import scipy as sy

def interpolate(dataFrame, method='values'):
    return dataFrame.interpolate(method=method)

def fill(dataFrame, method='', limit=1):
    if method=='':
        return dataFrame.fillna(0)
    else: return dataFrame.fillna(method=method, limit=limit)

def mask(dataFrame):
    return dataFrame.isnull()