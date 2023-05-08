# HSS407 -- Analysis of Fake news using Topic modeling

## Prerequisites

* Python
* Linux (should work on macOS too, but not tested, can't guarantee about Windows, it might not work)
* Requirements:
```
pip install -r requirements.txt
```

**Note**: if you want to get LDAvis visualizations, you need to install `pyLDAvis` from [source](https://github.com/bmabey/pyLDAvis), because its dependencies are outdated.

## Running

First run files to make LDA models:
```
python3 make_fake_models.py
python3 make_real_models.py
```

Then you can get the visualizations of metrics:
```
python3 calculate_metrics_fake.py
python3 calculate_metrics_real.py
```

To get wordclouds run
```
python3 make_wcs.py
```

To get topic visualizations first install `pyLDAvis`, then
```
pip3 install jupyter
python -m notebook
```
Then open `make_ldavis.ipynb` on jupyter notebook, and run cells one by one.
