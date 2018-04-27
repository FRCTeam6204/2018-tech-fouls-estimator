FRC Tech Foul Estimator
=======================

In each FRC round, two alliances of three teams each compete, and scores are recorded for each alliance.

The purpose of this analysis is to estimate the rate of tech fouls per team based on the data available *per alliance*.

The conclusion is that there is not really enough data in one tournament to generate accurate data, but the Least Squares method proved to be the most
versatile for generating rankings in tech foul order, although Maximum Likelihood was better at estimating actual numbers of tech fouls.

Details are documented in the `PDF Report <FoulEstimator.pdf>`_.


Program usage::

    usage: fouls_prediction.py [-h] [--data-file DATA_FILE] [--one-test]
                               [--long-test] [--competitions COMPETITIONS]
                               [--repeats REPEATS] [--real]

    optional arguments:
      -h, --help            show this help message and exit
      --data-file DATA_FILE
                            data file to load
      --one-test, -t        Run one test and display results
      --long-test, -l       Run multiple tests and summarise results
      --competitions COMPETITIONS, -c COMPETITIONS
                            Equivalent number of competitions to run test data on
      --repeats REPEATS, -i REPEATS
                            Number of repeats to run in long test
      --real, -r            Use the real data and summarise results
