import numpy as np

from src.models.gpt_wrapper import GPTModule

gpt_module = GPTModule()

def make_pred(stock_history, len_out=2, **kwargs):
    """
    returns Tuple[sample, median] for a given nparray or a Tuple[samples, medians] for an iterable over arrays
    """
    samples, medians = gpt_module(stock_history, len_out=len_out, **kwargs)
    return samples, medians

if __name__ == '__main__':
    # SANITY TEST

    example_hourly_stock_prices = np.array(
        [0, 631.0, 656.0, 650.0, 644.0, 634.0, 643.0, 651.0, 660.0, 661.0, 672.0, 683.0, 662.0, 654.0, 651.0, 641.0, 653.0, 641.0, 636.0, 635.0, 628.0, 633.0, 649.0, 650.0, 637.0, 633.0, 646.0, 645.0, 637.0, 632.0, 627.0, 625.0, 604.0, 604.0, 604.0, 601.0, 602.0, 595.0, 576.0, 563.0, 568.0, 587.0, 575.0, 572.0, 558.0, 558.0, 582.0, 581.0, 586.0, 583.0, 590.0, 591.0, 598.0, 584.0, 589.0, 600.0, 597.0, 589.0, 578.0, 566.0, 567.0, 558.0, 557.0, 573.0, 573.0, 562.0, 561.0, 565.0, 570.0, 565.0, 558.0, 547.0, 546.0, 526.0, 531.0, 538.0, 531.0, 534.0, 542.0, 528.0, 520.0, 518.0, 521.0, 505.0, 511.0, 520.0, 530.0, 538.0, 522.0, 530.0, 528.0, 520.0, 524.0, 514.0, 530.0, 547.0, 560.0, 549.0, 540.0, 540.0, 535.0, 544.0, 541.0, 530.0, 516.0, 513.0, 509.0, 508.0, 501.0, 498.0, 505.0, 494.5, 485.5],
        dtype=np.float64)  
    example_validate = np.array(
        [486.5, 498.5, 497.5, 491.0, 476.0],
        dtype=np.float64)
    samples, medians = make_pred(example_hourly_stock_prices, len_out=5)

    print(type(samples))

    print(type(medians))

    print('-----')
    print(samples)
    print('-----')
    print(medians)
    print('------')
    print(example_validate)
    err = np.array(medians) - example_validate
    print(err)