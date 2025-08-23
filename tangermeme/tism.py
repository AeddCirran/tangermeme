# tism.py

import torch
import itertools

from tqdm import trange


def _predict_grad(
    model,
    X,
    args=None,
    batch_size=32,
    target=None,
    dtype=None,
    device="cuda",
    verbose=False,
):
    """Make batched predictions in a memory-efficient manner.

    This function will take a PyTorch model and make predictions from it using
    the forward function, with optional additional arguments to the model. The
    additional arguments must have the same batch size as the examples, and the
    i-th example will be given to the model with the i-th index of each
    additional argument.

    Before starting predictions, the model is moved to the specified device. As
    predictions are being made, each batch is also moved to the specified
    device and then moved back to the CPU after predictions are made. Each batch
    is converted to the provided dtype if provided, keeping the original blob of
    examples in the original dtype. These features allow the function to work on
    massive data sets that do not fit in GPU memory. For example, the original
    sequences can be kept as 8-bit integers for compression and each batch will
    be upcast to the desired precision. If a single batch does not fit in memory,
    try lowering the batch size.


    Parameters
    ----------
    model: torch.nn.Module
            The PyTorch model to use to make predictions.

    X: torch.tensor, shape=(-1, len(alphabet), length)
            A one-hot encoded set of sequences to make predictions for.

    args: tuple or list or None, optional
            An optional set of additional arguments to pass into the model. If
            provided, each element in the tuple or list is one input to the model
            and the element must be formatted to be the same batch size as `X`. If
            None, no additional arguments are passed into the forward function.
            Default is None.

    func: function or None, optional
            A function to apply to a batch of predictions after they have been made.
            If None, do nothing to them. Default is None.

    batch_size: int, optional
            The number of examples to make predictions for at a time. Default is 32.

    target: int or slice or None
            If the user wants to subset to only some targets when calculating the
            average attribution across targets. If None, use all.

    dtype: str or torch.dtype or None, optional
            The dtype to use with mixed precision autocasting. If None, use the dtype of
            the *model*. This allows you to use int8 to represent large data sets and
            only convert batches to the higher precision, saving memory. Defailt is None.

    device: str or torch.device, optional
            The device to move the model and batches to when making predictions. If
            set to 'cuda' without a GPU, this function will crash and must be set
            to 'cpu'. Default is 'cuda'.

    verbose: bool, optional
            Whether to display a progress bar during predictions. Default is False.


    Returns
    -------
    y: torch.Tensor or list/tuple of torch.Tensors
            The output from the model for each input example. The precise format
            is determined by the model. If the model outputs a single tensor,
            y is a single tensor concatenated across all batches. If the model
            outputs multiple tensors, y is a list of tensors which are each
            concatenated across all batches.
    """

    model = model.to(device).eval()

    if dtype is None:
        try:
            dtype = next(model.parameters()).dtype
        except:
            dtype = torch.float32

    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if args is not None:
        for arg in args:
            if arg.shape[0] != X.shape[0]:
                raise ValueError(
                    "Arguments must have the same first " + "dimension as X"
                )

    ###

    y = []
    with torch.autograd.set_grad_enabled(True):
        batch_size = min(batch_size, X.shape[0])

        for start in trange(0, X.shape[0], batch_size, disable=not verbose):
            end = start + batch_size
            X_ = X[start:end].type(dtype).to(device).requires_grad_()

            if X_.shape[0] == 0:
                continue

            with torch.autocast(device_type=device, dtype=dtype):
                if args is not None:
                    args_ = [a[start:end].type(dtype).to(device) for a in args]
                    y_ = model(X_, *args_)
                else:
                    y_ = model(X_)

                y_target = y_[:, target]

                if len(y_target.shape) > 1:
                    y_target = torch.mean(
                        y_target, dim=tuple(range(1, len(y_target.shape)))
                    )

                y_target_like = torch.ones_like(y_target)
                y_grad = (
                    torch.autograd.grad(y_target, X_, grad_outputs=y_target_like)[0]
                    .detach()
                    .cpu()
                )

            y.append(y_grad)

    return torch.cat(y)


def _attribution_score(y_hat_grad):
    """An internal function for calculating the TISM attributions.

    This function, which is meant to be used for TISM, will take in the
    predictions before and after substitutions and a target -- which can be
    None -- and return the position-normalized differences. Specifically,
    for each example, the differences in prediction will be normalized by
    subtracting out the per-position average, and then averaged across
    all tasks if target is None.


    Parameters
    ----------
    y_hat: torch.Tensor, shape=(-1, len(alphabet), length, n_targets)
            Model predictions for each example for each substitution.
    """

    y_hat_grad = torch.mean(y_hat_grad, dim=1)
    attr = y_hat_grad
    attr -= torch.sum(attr, dim=1, keepdims=True)

    return attr


def _edit_distance_one(X, start, end):
    """An internal function for generating all sequences of edit distance 1

    This internal function, which is meant to be used for ISM, will take in a
    one-hot encoded sequence and return all sequences that have an edit distance
    of one.


    Parameters
    ----------
    X: torch.Tensor, shape=(len(alphabet), sequence_length)
            A single one-hot encoded sequence.

    start: int
            The first nucleotide to begin making edits on, inclusive.

    end: int
            The end of the span. Edits are not made on this nucleotide at this
            index. Can be negative indexes.


    Returns
    -------
    X_: torch.Tensor, shape=(length*len(alphabet), len(alphabet), length)
            All one-hot encoded sequences that have an edit distance of 1 from the
            original sequence.
    """

    end = end if end >= 0 else X.shape[-1] + 1 + end
    X_ = X.repeat((end - start), 1, 1)

    coords = range(start, end)
    for i, k in enumerate(coords):
        X_[i, :, k] = 0

    return X_


def tism(
    model,
    X,
    args=None,
    start=0,
    end=-1,
    batch_size=32,
    target=None,
    hypothetical=False,
    raw_outputs=False,
    dtype=None,
    device="cuda",
    verbose=False,
):
    """Performs in-silico saturation mutagenesis on a set of sequences.

    This function will perform in-silico saturation mutagenesis on a set of
    sequences and return the predictions on the original sequences and each
    of the sequences with an edit distance of one on them.

    By default, this function will aggregate these predictions into an
    attribution value. This aggregation involves taking the Euclidean distance
    between the predictions before and after the substitutions and Z-score
    normalizing them across the entire example. However, this assumes that
    the model returns only a single tensor. This tensor can have multiple
    outputs, e.g., be of shape (batch_size, n_targets) where n_targets > 1,
    but the model cannot return multiple tensors.

    If you simply want the predictions before and after the substitutions
    without the method turning those into attributions because, perhaps, you
    want to define your own aggregation method, you can use `raw_outputs=True`.


    Parameters
    ----------
    model: torch.nn.Module
            The PyTorch model to use to make predictions.

    X: torch.tensor, shape=(-1, len(alphabet), length)
            A set of one-hot encoded sequences to calculate attribution values
            for.

    args: tuple or None, optional
            An optional set of additional arguments to pass into the model. If
            provided, each element in the tuple or list is one input to the model
            and the element must be formatted to be the same batch size as `X`. If
            None, no additional arguments are passed into the forward function.
            Default is None.

    start: int, optional
            The start of where to begin making perturbations to the sequence.
            Default is 0.

    end: int, optional
            The end of where to to make perturbations to the sequence. If end is
            positive, it is non-inclusive. If end is negative, it is inclusive.
            Default is -1, meaning the entire sequence.

    batch_size: int, optional
            The number of examples to make predictions for at a time. Default is 32.

    target: int or slice or None, optional
            Whether to focus on a single output/slice of outputs from the model
            when calculating attributions rather than the entire set of outputs.
            If None, use all targets when calculating distances. Default is None.

    hypothetical: bool, optional
            Whether to return attributions for all possible characters at each
            position or only for the character that is actually at the sequence.
            Only matters when `raw_outputs=False`. Default is False.

    raw_outputs: bool, optional
            Whether to return the raw outputs from the method -- in this case,
            the predictions from the reference sequence and from each of the
            perturbations -- or the processed attribution values. Default is False.

    dtype: str or torch.dtype or None, optional
            The dtype to use with mixed precision autocasting. If None, use the dtype of
            the *model*. This allows you to use int8 to represent large data sets and
            only convert batches to the higher precision, saving memory. Defailt is None.

    device: str or torch.device, optional
            The device to move the model and batches to when making predictions. If
            set to 'cuda' without a GPU, this function will crash and must be set
            to 'cpu'. Default is 'cuda'.

    verbose: bool, optional
            Whether to display a progress bar during predictions. Default is False.


    Returns
    -------
    attr: torch.Tensor
            Processed attribution values as the z-score normalized difference
            between the difference in predictions for the original sequence and
            the perturbed sequences.

    -- or, if raw_outputs=True --

    y_hat: torch.Tensor or list/tuple of torch.Tensors
            The outputs from the model for each of the perturbed sequences.
    """

    if end < 0:
        end = X.shape[-1] + 1 + end

    y_hat_grad = []
    for i in range(X.shape[0]):
        X_ = _edit_distance_one(X[i], start, end)

        if args is not None:
            args_ = tuple(
                a[i].repeat(X_.shape[0], *(1 for _ in a[i].shape)) for a in args
            )
        else:
            args_ = None

        y_hat_grad_ = _predict_grad(
            model,
            X_,
            args=args_,
            batch_size=batch_size,
            target=target,
            dtype=dtype,
            device=device,
            verbose=verbose,
        )

        y_hat_grad.append(y_hat_grad_)

    y_hat_grad = torch.stack(y_hat_grad).reshape(
        X.shape[0], end - start, *y_hat_grad_.shape[1:]
    )

    if raw_outputs == False:
        attr = _attribution_score(y_hat_grad)

        return X * attr if hypothetical == False else attr

    return y_hat_grad
