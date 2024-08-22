from ldctbench.hub import Methods, load_model


def test_laod_cnn10():
    net = load_model(Methods.CNN10)
    assert net


def test_laod_redcnn():
    net = load_model(Methods.REDCNN)
    assert net


def test_laod_wganvgg():
    net = load_model(Methods.WGANVGG)
    assert net


def test_laod_resnet():
    net = load_model(Methods.RESNET)
    assert net


def test_laod_qae():
    net = load_model(Methods.QAE)
    assert net


def test_laod_dugan():
    net = load_model(Methods.DUGAN)
    assert net


def test_laod_transct():
    net = load_model(Methods.TRANSCT)
    assert net


# Do not test Bilateral here as it needs a separate GPU-only python package installed


def test_if_eval_true_then_model_in_eval():
    net = load_model(Methods.RESNET, eval=True)
    assert not net.training


def test_if_eval_false_then_model_in_training():
    net = load_model(Methods.RESNET, eval=False)
    assert net.training
