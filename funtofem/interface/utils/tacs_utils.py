__all__ = ["_resolve_ks_options"]


def _resolve_ks_options(options: dict) -> dict:
    """
    Normalize KS function options dict for TACS compatibility:
      - Promotes deprecated lowercase 'ksweight' key to camelCase 'ksWeight'
      - Converts 'aggregation_type' string to 'ksAggregationType' enum (TACS >= PR #439)
        or to the 'ftype' string kwarg for older TACS versions.
    """

    import warnings

    options = dict(options)  # don't mutate the original

    # Handle deprecated lowercase 'ksweight' key
    if "ksweight" in options:
        warnings.warn(
            "option key 'ksweight' is deprecated, use 'ksWeight' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        options["ksWeight"] = options.pop("ksweight")

    # Convert 'aggregation_type' string to whatever TACS expects
    if "aggregation_type" in options:
        agg_str = options.pop("aggregation_type")

        try:
            from tacs.functions import KSAggregationType

            _agg_map = {
                "continuous": KSAggregationType.KS_CONTINUOUS,
                "discrete": KSAggregationType.KS_DISCRETE,
                "pnorm-continuous": KSAggregationType.PNORM_CONTINUOUS,
                "pnorm-discrete": KSAggregationType.PNORM_DISCRETE,
            }
            agg_type = _agg_map.get(agg_str.lower())
            if agg_type is None:
                raise ValueError(
                    f"Unknown aggregation_type '{agg_str}'. "
                    f"Valid options are: {list(_agg_map.keys())}"
                )
            options["ksAggregationType"] = agg_type

        except ImportError:
            # Pre-PR 439 TACS: pass as the old 'ftype' string kwarg
            options["ftype"] = agg_str

    return options
