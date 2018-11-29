# rl-playlist-generation

Original implementation of ["Automatic, Personalized, and Flexible Playlist Generation using Reinforcement Learning"](http://ismir2018.ircam.fr/doc/pdfs/18_Paper.pdf).

## Dependencies

* python3

You can check and install other dependencies in requirements.txt

```shell
$ pip3 install -r requirements.txt
# to install TensorFlow, you can refer to https://www.tensorflow.org/install/
```

## Files you should prepare

### Pretrain

The following are files you should prepare to train this playlist generation model.
Besides, you can check sample files we prepare as a reference of formats.

#### data/raw/raw_data.txt
```
[date_of_created_playlist] [song_id1] [song_id2] ....
```

#### data/embedding.txt
```
# embedding: [value1 value2 value3 ...]
[song_id] [value1] [value2] [value3] ...
```

### Reinforcement Learning

#### data/metrics.txt
```
[song_id] [popularity_score] [artist_id] [release_date]
```

### Test

#### results/in.txt

```
[seed_song_id]
```

## Usage

You can add argument `--debug 1` for each mode to check everything is fine
before you prepare your own data.

### Prepare data
```shell
# files mentioned above should be created first
$ ./prepare_data.sh
```

### Pretrain

```shell
$ python3 main.py --mode pretrain [--debug 1]
```

### Reinforcement Learning

```shell
$ python3 main.py --mode rl [--debug 1]
```

### Test

```shell
# input file: results/in.txt
# output file: results/out.txt
$ python3 main.py --mode test [--debug 1]
```

## Other Arguments

If you would like some different settings for this model, you can refer to lib/config.py.

