import jittor as jt


def validate(model, batch, Ke, F):
    model.eval()
    with jt.no_grad():
        displacement, residual = model(batch, Ke, F)
        residual_norm = jt.sqrt((residual * residual).sum())
    model.train()
    return {
        "displacement": displacement,
        "residual": residual,
        "residual_norm": residual_norm,
    }
