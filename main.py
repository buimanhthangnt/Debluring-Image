from model import AutoEncoderDecoder

auto_enc_dec = AutoEncoderDecoder(56, 46, 1)
auto_enc_dec.build()
auto_enc_dec.load_weights()

# auto_enc_dec.sample()
auto_enc_dec.test()
