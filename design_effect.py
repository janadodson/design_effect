"""
Functions related to statistical hypothesis testing of clustered and/or weighted data.

Author : Jana Dodson
Email : janadodson1@gmail.com
Version : 1.0

Dependencies
------------
numpy : version 1.11.3
pandas : version 0.19.2
"""


import numpy as np
import pandas as pd


__all__ = ['ss', 'df', 'ms', 'avg_sz', 'icc', 'de', 'effn', 'se']


def check_array_dims(array_list):
    """
    Check that all input arrays are 1D and of the same length. 

    Parameters
    ----------
    array_list : list
        List of input arrays to check.

    Returns
    -------
    None
    """

    # Iterate through arrays
    for a in array_list:

        # Check that array is 1 dimensional
        if a.ndim != 1:
            raise Exception('Input arrays are not all 1 dimensional.')

        # Check that array has the same shape as the first array in the list
        if a.shape != array_list[0].shape:
            raise Exception('Input arrays are not all the same shape.')


def ss(input_level, component='total', **kwargs):
    """
    Compute within-cluster, between-cluster, or total sum of squares.

    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    component : string, default = 'total'
        Component of sum of squares to return.
        * 'within': Within-cluster sum of squares.
        * 'between': Between-cluster sum of squares.
        * 'total': Total sum of squares.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.

    Returns
    -------
    ss : numpy.float64
        Within-cluster, between-cluster, or total sum of squares.
    """
    
    # Observation-level input arrays
    if input_level == 'observation':
        
        # Convert input arrays to numpy arrays
        # Defaults:
        # All observations are equally weighted
        # All observations are in their own cluster
        x = np.asanyarray(kwargs.get('x'))
        wts = np.asanyarray(kwargs.get('wts', np.ones(x.size)))
        cluster_labels = np.asanyarray(kwargs.get('cluster_labels', np.arange(x.size)))

        # Check input array dimensions
        check_array_dims([x, wts, cluster_labels])
        
        # Convert input arrays to pandas series with cluster labels as index
        x = pd.Series(x, index=cluster_labels)
        wts = pd.Series(wts, index=cluster_labels)
        
        # Aggregate to cluster level
        avg = (x * wts).groupby(level=0).sum() / wts.groupby(level=0).sum()
        var = (x**2 * wts).groupby(level=0).sum() / wts.groupby(level=0).sum() - avg**2
        sum_wts = wts.groupby(level=0).sum()
    
    # Cluster-level input arrays
    elif input_level == 'cluster':
        
        # Convert input arrays to numpy arrays
        avg = np.asanyarray(kwargs.get('avg'))
        var = np.asanyarray(kwargs.get('var'))
        sum_wts = np.asanyarray(kwargs.get('sum_wts'))

        # Check input array dimensions
        check_array_dims([avg, var, sum_wts])
        
    else:
        raise Exception("Parameter 'input_level' is restricted to the following values: 'cluster' or 'observation'.")

    # Calculate within- and between-cluster sum of squares
    ss_wn = (sum_wts * var).sum()
    ss_bw = (sum_wts * avg**2).sum() - sum_wts.sum() * np.average(a=avg, weights=sum_wts)**2

    # Return sum of squares
    if component == 'within':
        return ss_wn
    elif component == 'between':
        return ss_bw
    elif component == 'total':
        return ss_bw + ss_wn
    else:
        raise Exception("Parameter 'component' is restricted to the following values: 'within', 'between', or 'total'.")  


def df(input_level, component='total', **kwargs):
    """
    Compute within-cluster, between-cluster, or total degrees of 
    freedom.

    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    component : string, default = 'total'
        Component of degrees of freedom to return.
        * 'within': Within-cluster degrees of freedom.
        * 'between': Between-cluster degrees of freedom.
        * 'total': Total degrees of freedom.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.

    Returns
    -------
    df : numpy.float64
        Within-cluster, between-cluster, or total degrees of freedom.
    """
    
    # Observation-level input arrays
    if input_level == 'observation':
        
        # Convert input arrays to numpy arrays
        # Defaults:
        # All observations are equally weighted
        # All observations are in their own cluster
        x = np.asanyarray(kwargs.get('x'))
        wts = np.asanyarray(kwargs.get('wts', np.ones(x.size)))
        cluster_labels = np.asanyarray(kwargs.get('cluster_labels', np.arange(x.size)))

        # Check input array dimensions
        check_array_dims([x, wts, cluster_labels])

        # Convert input arrays to pandas series with cluster labels as index
        x = pd.Series(x, index=cluster_labels)
        wts = pd.Series(wts, index=cluster_labels)
        
        # Aggregate to cluster level
        sum_wts = wts.groupby(level=0).sum()
        sum_sqd_wts = (wts**2).groupby(level=0).sum()
            
    # Cluster-level input arrays
    elif input_level == 'cluster':
        
        # Convert input arrays to numpy arrays
        sum_wts = np.asanyarray(kwargs.get('sum_wts'))
        sum_sqd_wts = np.asanyarray(kwargs.get('sum_sqd_wts'))

        # Check input array dimensions
        check_array_dims([sum_wts, sum_sqd_wts])
        
    else:
        raise Exception("Parameter 'input_level' is restricted to the following values: 'cluster' or 'observation'.")

    # Calculate within- and between-cluster degrees of freedom
    df_wn = (sum_wts - sum_sqd_wts / sum_wts).sum()
    df_bw = (sum_sqd_wts / sum_wts).sum() - sum_sqd_wts.sum() / sum_wts.sum()

    # Return degrees of freedom
    if component == 'within':
        return df_wn
    elif component == 'between':
        return df_bw
    elif component == 'total':
        return df_bw + df_wn
    else:
        raise Exception("Parameter 'component' is restricted to the following values: 'within', 'between', or 'total'.")


def ms(input_level, component='total', **kwargs):
    """
    Compute within-cluster, between-cluster, or total mean squares.

    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    component : string, default = 'total'
        Component of mean squares to return.
        * 'within': Within-cluster mean squares.
        * 'between': Between-cluster mean squares.
        * 'total': Total mean squares.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.

    Returns
    -------
    ms : numpy.float64
        Within-cluster, between-cluster, or total mean squares.
    """
    
    ss_comp = ss(input_level, component, **kwargs)
    df_comp = df(input_level, component, **kwargs)

    if df_comp == 0:

        # If all observations are in their own cluster, within-cluster mean squares is 0
        if component == 'within':
            return 0

        # If there is only one cluster, throw error
        else:
            raise Exception("More than 1 cluster required.")

    else:
        return ss_comp / df_comp


def avg_sz(input_level, **kwargs):
    """
    Compute average cluster size.
    
    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.

    Returns
    -------
    avg_sz : numpy.float64
        Average cluster size.
    """
    
    # Observation-level input arrays
    if input_level == 'observation':
        
        # Convert input arrays to numpy arrays
        # Defaults:
        # All observations are equally weighted
        # All observations are in their own cluster
        x = np.asanyarray(kwargs.get('x'))
        wts = np.asanyarray(kwargs.get('wts', np.ones(x.size)))
        cluster_labels = np.asanyarray(kwargs.get('cluster_labels', np.arange(x.size)))

        # Check input array dimensions
        check_array_dims([x, wts, cluster_labels])
        
        # Convert input arrays to pandas series with cluster labels as index
        x = pd.Series(x, index=cluster_labels)
        wts = pd.Series(wts, index=cluster_labels)
        
        # Aggregate to cluster level
        sum_wts = wts.groupby(level=0).sum()
        sum_sqd_wts = (wts**2).groupby(level=0).sum()
            
    # Cluster-level input arrays
    elif input_level == 'cluster':
        
        # Convert input arrays to numpy arrays
        sum_wts = np.asanyarray(kwargs.get('sum_wts'))
        sum_sqd_wts = np.asanyarray(kwargs.get('sum_sqd_wts'))

        # Check input array dimensions
        check_array_dims([sum_wts, sum_sqd_wts])
        
    else:
        raise Exception("Parameter 'input_level' is restricted to the following values: 'cluster' or 'observation'.")

    # Calculate average cluster size
    return sum_wts.sum() / (sum_sqd_wts / sum_wts).sum()


def icc(input_level, **kwargs):
    """
    Compute intraclass correlation coefficient.
    
    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.

    Returns
    -------
    icc : numpy.float64
        Intraclass correlation coefficient.
    """
    
    # Calculate between- and within-cluster mean squares
    ms_bw = ms(input_level, 'between', **kwargs)
    ms_wn = ms(input_level, 'within', **kwargs)
    
    # Calculate intraclass correlation coefficient
    return (ms_bw - ms_wn) / (ms_bw + (avg_sz(input_level, **kwargs) - 1) * ms_wn)


def de(input_level, component='total', icc_floor=0., **kwargs):
    """
    Compute design effect.
    
    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    component : string, default = 'total'
        Component of design effect to return.
        * 'weighting': Design effect due to weighting.
        * 'clustering': Design effect due to clustering.
        * 'total': Total design effect.
    icc_floor : float, default = 0.
        Artificial floor for ICC. Default is set to 0 because although 
        an ICC between -1 and 0 is mathematically possible, it is 
        practically meaningless.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.
    n : array-like, shape = (n_clusters,)
        1D array containing the number of observations within each 
        cluster.

    Returns
    -------
    de : numpy.float64
        Design effect.
    """

    # Observation-level input arrays
    if input_level == 'observation':
        
        # Convert input arrays to numpy arrays
        # Defaults:
        # All observations are equally weighted
        # All observations are in their own cluster
        x = np.asanyarray(kwargs.get('x'))
        wts = np.asanyarray(kwargs.get('wts', np.ones(x.size)))
        cluster_labels = np.asanyarray(kwargs.get('cluster_labels', np.arange(x.size)))

        # Check input array dimensions
        check_array_dims([x, wts, cluster_labels])
        
        # Convert input arrays to pandas series with cluster labels as index
        x = pd.Series(x, index=cluster_labels)
        wts = pd.Series(wts, index=cluster_labels)
        
        # Aggregate to cluster level
        sum_wts = wts.groupby(level=0).sum()
        sum_sqd_wts = (wts**2).groupby(level=0).sum()
        n = x.groupby(level=0).count()
            
    # Cluster-level input arrays
    elif input_level == 'cluster':
        
        # Convert input arrays to numpy arrays
        sum_wts = np.asanyarray(kwargs.get('sum_wts'))
        sum_sqd_wts = np.asanyarray(kwargs.get('sum_sqd_wts'))
        n = np.asanyarray(kwargs.get('n'))

        # Check input array dimensions
        check_array_dims([sum_wts, sum_sqd_wts, n])
        
    else:
        raise Exception("Parameter 'input_level' is restricted to the following values: 'cluster' or 'observation'.")
    
    # Calculate weighting and clustering design effect
    de_weighting = n.sum() / (sum_wts.sum()**2 / sum_sqd_wts.sum())
    de_clustering = 1 + np.maximum(icc(input_level, **kwargs), icc_floor) * (avg_sz(input_level, **kwargs) - 1)
    
    # Return design effect
    if component == 'weighting':
        return de_weighting
    elif component == 'clustering':
        return de_clustering
    elif component == 'total':
        return de_weighting * de_clustering
    else:
        raise Exception("Parameter 'component' is restricted to the following values: 'weighting', 'clustering', or 'total'.")


def effn(input_level, icc_floor=0., **kwargs):
    """
    Compute effective sample size.
    
    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    icc_floor : float, default = 0.
        Artificial floor for ICC. Default is set to 0 because although 
        an ICC between -1 and 0 is mathematically possible, it is 
        practically meaningless.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.
    n : array-like, shape = (n_clusters,)
        1D array containing the number of observations within each 
        cluster.

    Returns
    -------
    effn : numpy.float64
        Effective sample size.
    """

    # Observation-level input arrays
    if input_level == 'observation':
        
        # Convert input arrays to numpy arrays
        # Defaults:
        # All observations are in their own cluster
        x = np.asanyarray(kwargs.get('x'))
        cluster_labels = np.asanyarray(kwargs.get('cluster_labels', np.arange(x.size)))

        # Check input array dimensions
        check_array_dims([x, cluster_labels])
        
        # Convert input arrays to pandas series with cluster labels as index
        x = pd.Series(x, index=cluster_labels)
        
        # Aggregate to cluster level
        n = x.groupby(level=0).count()
            
    # Cluster-level input arrays
    elif input_level == 'cluster':
        
        # Convert input arrays to numpy arrays
        n = np.asanyarray(kwargs.get('n'))

        # Check input array dimensions
        check_array_dims([n])
        
    else:
        raise Exception("Parameter 'input_level' is restricted to the following values: 'cluster' or 'observation'.")
    
    # Return effective sample size
    return n.sum() / de(input_level, 'total', **kwargs)


def se(input_level, icc_floor=0., **kwargs):
    """
    Compute standard error.
    
    Parameters
    ----------
    input_level : string
        Aggregation level of input arrays.
        * 'observation': Input arrays contain one element per 
        observation.
        * 'cluster': Input arrays contain one element per cluster.
    icc_floor : float, default = 0.
        Artificial floor for ICC. Default is set to 0 because although 
        an ICC between -1 and 0 is mathematically possible, it is 
        practically meaningless.

    Keyword Arguments - when input_level = 'observation'
    ----------------------------------------------------
    x : array-like, shape = (n_observations,)
        1D array containing the value for each observation.
    wts : array-like, optional, shape = (n_observations,)
        1D array containing the weight for each observation. When 
        unspecified, observations are assumed to be weighted equally.
    cluster_labels : array-like, optional, shape = (n_observations,)
        1D array containing the cluster label for each observation. When 
        unspecified, observations are assumed to all be in their own 
        cluster. 

    Keyword Arguments - when input_level = 'cluster'
    ------------------------------------------------  
    avg : array-like, shape = (n_clusters,)
        1D array containing the average within each cluster.
    var : array-like, shape = (n_clusters,)
        1D array containing the population variance within each cluster.
    sum_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of observation weights within each 
        cluster or, when observations are unweighted, the number of 
        observations within each cluster.
    sum_sqd_wts : array-like, shape = (n_clusters,)
        1D array containing the sum of squared observation weights 
        within each cluster or, when observations are unweighted, the 
        number of observations within each cluster.
    n : array-like, shape = (n_clusters,)
        1D array containing the number of observations within each 
        cluster.

    Returns
    -------
    se : numpy.float64
        Standard error.
    """

    # Return standard error
    return np.sqrt(ms(input_level, 'total', **kwargs) / effn(input_level, **kwargs))