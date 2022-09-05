""" from https://github.com/keithito/tacotron """


## for Persian Language
_pad = "_"
persian_phonemes = ['U', 'Q', 'G', 'AA', 'V', 'N', 'CH', 'R', 'KH', 'B', 'Z', 'SH', 'O', 'A', 'E', 'ZH', 'H', 'SIL', 'AH', 'S', 'D', 'J', 'L', 'F', 'K', 'I', 'T', 'P', 'M', 'Y']
persian_phonemes += ['?', '!', '.', ',', ';', ':']
persian_symbols = [_pad] + persian_phonemes