def get_curriculum_value(base_value: float, epoch: int, curriculum_epochs: int,
                         min_value: float = 0.0) -> float:
    if curriculum_epochs <= 0:
        return base_value
    if epoch >= curriculum_epochs:
        return base_value
    frac = (epoch + 1) / curriculum_epochs
    return min_value + (base_value - min_value) * frac
