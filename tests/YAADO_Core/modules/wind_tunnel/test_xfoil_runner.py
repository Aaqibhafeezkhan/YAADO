"""Hand-check tests for the Ackeret supersonic thin-airfoil polar."""

import pytest

from YAADO_Core.modules.wind_tunnel.methods.xfoil.xfoil_runner import ackeret_polar


@pytest.mark.parametrize(
    ("mach", "alpha_deg", "t_c", "expected_cl", "expected_cd"),
    [
        # Reference values independently evaluated from Anderson's Ackeret formulas.
        (2.0, 5.0, 0.1, 0.20153326269269087, 0.040528316392067186),
        # At zero incidence, lift is zero but thickness wave drag remains.
        (3.0, 0.0, 0.1, 0.0, 0.014048571663723437),
        # Negative incidence reverses lift while drag remains positive.
        (2.5, -4.0, 0.06, -0.1218758614839569, 0.01477817377181453),
    ],
)
def test_ackeret_polar_matches_hand_checked_reference_values(
    mach: float,
    alpha_deg: float,
    t_c: float,
    expected_cl: float,
    expected_cd: float,
) -> None:
    """Compare lift and drag with independently calculated textbook values."""
    cl, cd = ackeret_polar(mach, alpha_deg, t_c)

    assert cl == pytest.approx(expected_cl, rel=1e-9, abs=1e-12)
    assert cd == pytest.approx(expected_cd, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("mach", [0.8, 1.0])
def test_ackeret_polar_rejects_mach_at_or_below_unity(mach: float) -> None:
    """Ackeret theory must reject subsonic flow and the Mach 1 singularity."""
    with pytest.raises(ValueError, match="Mach > 1"):
        ackeret_polar(mach, alpha_deg=5.0, t_c=0.1)
