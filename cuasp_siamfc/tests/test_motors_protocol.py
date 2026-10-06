"""Pure unit tests for motors.py's PCMM wire-protocol parsers/formatters.

None of this has any test coverage today despite being the one place a
malformed PCMM reply could cause a wrong PCMMStatus to be trusted -- these
are cheap, deterministic string-parsing tests with no socket I/O.
"""

import math

import pytest

from suncubes import motors


# ---------------------------------------------------------------------------
# _clip_range / _parse_bool / _normalise_encoder_axes
# ---------------------------------------------------------------------------


def test_clip_range_within_bounds_is_unchanged():
    assert motors._clip_range(5.0, -10.0, 10.0) == 5.0


def test_clip_range_clamps_above_and_below():
    assert motors._clip_range(15.0, -10.0, 10.0) == 10.0
    assert motors._clip_range(-15.0, -10.0, 10.0) == -10.0


def test_clip_range_rejects_non_finite():
    with pytest.raises(ValueError):
        motors._clip_range(math.nan, -10.0, 10.0)


def test_parse_bool_accepts_true_false_and_numeric():
    assert motors._parse_bool("TRUE") is True
    assert motors._parse_bool("true") is True
    assert motors._parse_bool("1") is True
    assert motors._parse_bool("FALSE") is False
    assert motors._parse_bool("0") is False


def test_parse_bool_rejects_garbage():
    with pytest.raises(ValueError):
        motors._parse_bool("maybe")


def test_normalise_encoder_axes_accepts_0_and_1():
    assert motors._normalise_encoder_axes([0, 1]) == (0, 1)
    assert motors._normalise_encoder_axes([1]) == (1,)


def test_normalise_encoder_axes_rejects_out_of_range():
    with pytest.raises(ValueError):
        motors._normalise_encoder_axes([2])


def test_normalise_encoder_axes_rejects_duplicates():
    with pytest.raises(ValueError):
        motors._normalise_encoder_axes([0, 0])


def test_normalise_encoder_axes_rejects_empty():
    with pytest.raises(ValueError):
        motors._normalise_encoder_axes([])


# ---------------------------------------------------------------------------
# _parse_encoder_reply
# ---------------------------------------------------------------------------


def test_parse_encoder_reply_two_axes():
    values = motors._parse_encoder_reply("enc[0,1]=10.5,-3.25\n", (0, 1))
    assert values == (10.5, -3.25)


def test_parse_encoder_reply_reorders_to_match_requested_axes():
    values = motors._parse_encoder_reply("enc[1,0]=-3.25,10.5\n", (0, 1))
    assert values == (10.5, -3.25)


def test_parse_encoder_reply_single_axis():
    assert motors._parse_encoder_reply("enc[0]=1.0\n", (0,)) == (1.0,)


def test_parse_encoder_reply_missing_requested_axis_raises():
    with pytest.raises(ValueError):
        motors._parse_encoder_reply("enc[0]=1.0\n", (0, 1))


def test_parse_encoder_reply_malformed_raises():
    with pytest.raises(ValueError):
        motors._parse_encoder_reply("not an encoder reply", (0,))


# ---------------------------------------------------------------------------
# _parse_status_reply -- both schemas
# ---------------------------------------------------------------------------


def test_parse_status_reply_15_field_position_v1_schema():
    reply = "status=2,42,TRUE,TRUE,TRUE,1.5,-0.5,10.0,5.0,10.1,4.9,0.01,0.005,FALSE,FALSE\n"
    status = motors._parse_status_reply(reply)
    assert status.controller_state == 2
    assert status.position_reference_sequence == 42
    assert status.position_reference_fresh is True
    assert status.position_reference_valid is True
    assert status.tcp_connected is True
    assert status.axis1_velocity_command_deg_s == 1.5
    assert status.axis2_velocity_command_deg_s == -0.5
    assert status.axis1_reference_deg == 10.0
    assert status.axis2_reference_deg == 5.0
    assert status.axis1_actual_deg == 10.1
    assert status.axis2_actual_deg == 4.9
    assert status.reference_dt_s == 0.01
    assert status.reference_age_s == 0.005
    assert status.position_reference_clamped is False
    assert status.controller_fault is False
    assert status.status_schema == motors.PCMM_STATUS_SCHEMA_POSITION_V1


def test_parse_status_reply_12_field_legacy_schema():
    reply = "status=1,7,TRUE,FALSE,TRUE,0.2,-0.3,0,0,0,0.02,0.01\n"
    status = motors._parse_status_reply(reply)
    assert status.controller_state == 1
    assert status.position_reference_sequence == 7
    assert status.position_reference_fresh is True
    assert status.position_reference_valid is True  # fresh and not timed_out
    assert status.tcp_connected is True
    assert status.axis1_velocity_command_deg_s == 0.2
    assert status.axis2_velocity_command_deg_s == -0.3
    assert math.isnan(status.axis1_reference_deg)
    assert math.isnan(status.axis2_actual_deg)
    assert status.reference_dt_s == 0.02
    assert status.reference_age_s == 0.01
    assert status.status_schema == motors.PCMM_STATUS_SCHEMA_LEGACY_V6


def test_parse_status_reply_legacy_schema_marks_invalid_when_timed_out():
    reply = "status=1,7,TRUE,TRUE,TRUE,0.2,-0.3,0,0,0,0.02,0.01\n"  # timed_out=TRUE
    status = motors._parse_status_reply(reply)
    assert status.position_reference_fresh is True
    assert status.position_reference_valid is False


def test_parse_status_reply_wrong_field_count_raises():
    with pytest.raises(ValueError):
        motors._parse_status_reply("status=1,2,3\n")


def test_parse_status_reply_missing_prefix_raises():
    with pytest.raises(ValueError):
        motors._parse_status_reply("2,42,TRUE,TRUE,TRUE,1.5,-0.5,10.0,5.0,10.1,4.9,0.01,0.005,FALSE,FALSE\n")
