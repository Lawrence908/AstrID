"""Tests for survey-specific adapters."""

from uuid import uuid4

import pytest

from src.domains.observations.adapters import (
    HSTAdapter,
    JWSTAdapter,
    LSSTAdapter,
    SDSSAdapter,
)
from src.domains.observations.schema import ObservationCreate


class TestHSTAdapter:
    """Test suite for HST data adapter."""

    @pytest.fixture
    def adapter(self):
        """Create HST adapter for testing."""
        return HSTAdapter()

    @pytest.fixture
    def valid_hst_data(self):
        """Valid HST observation data."""
        return {
            "obs_id": "hst_12345_01_acs_wfc_f606w",
            "s_ra": 210.8023,
            "s_dec": 54.3489,
            "t_min": 58849.5,  # MJD
            "filters": "F606W",
            "t_exptime": 507.0,
            "instrument_name": "ACS",
            "detector": "WFC",
            "obs_collection": "HST",
            "target_name": "NGC5194",
            "proposal_id": "12345",
            "dataURL": "https://mast.stsci.edu/api/v0.1/Download/file?uri=test.fits",
        }

    @pytest.mark.asyncio
    async def test_normalize_observation_data_success(self, adapter, valid_hst_data):
        """Test successful HST data normalization."""
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(valid_hst_data, survey_id)

        assert isinstance(result, ObservationCreate)
        assert result.survey_id == survey_id
        assert result.observation_id == "hst_12345_01_acs_wfc_f606w"
        assert result.ra == 210.8023
        assert result.dec == 54.3489
        assert result.filter_band == "V"  # F606W mapped to V
        assert result.exposure_time == 507.0
        assert result.airmass is None  # HST is in space
        assert result.pixel_scale == 0.05  # ACS/WFC pixel scale

    @pytest.mark.asyncio
    async def test_normalize_observation_data_missing_fields(self, adapter):
        """Test normalization with missing required fields."""
        invalid_data = {
            "obs_id": "test",
            # Missing s_ra, s_dec, etc.
        }
        survey_id = uuid4()

        with pytest.raises(ValueError, match="HST data validation failed"):
            await adapter.normalize_observation_data(invalid_data, survey_id)

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, adapter, valid_hst_data):
        """Test HST metadata extraction."""
        metadata = await adapter.extract_metadata(valid_hst_data)

        assert metadata["instrument"] == "ACS"
        assert metadata["detector"] == "WFC"
        assert metadata["proposal_id"] == "12345"
        assert metadata["target_name"] == "NGC5194"
        assert metadata["mission"] == "HST"
        assert metadata["observatory"] == "Hubble Space Telescope"

    @pytest.mark.asyncio
    async def test_validate_survey_specific_data_success(self, adapter, valid_hst_data):
        """Test HST data validation success."""
        is_valid = await adapter.validate_survey_specific_data(valid_hst_data)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_survey_specific_data_invalid_coordinates(
        self, adapter, valid_hst_data
    ):
        """Test HST data validation with invalid coordinates."""
        valid_hst_data["s_ra"] = 400.0  # Invalid RA

        with pytest.raises(ValueError, match="Invalid RA"):
            await adapter.validate_survey_specific_data(valid_hst_data)

    def test_map_filter_band(self, adapter):
        """Test HST filter mapping."""
        assert adapter.map_filter_band("F606W") == "V"
        assert adapter.map_filter_band("F814W") == "I"
        assert adapter.map_filter_band("UNKNOWN") == "UNKNOWN"

    def test_get_supported_filters(self, adapter):
        """Test getting supported HST filters."""
        filters = adapter.get_supported_filters()
        assert "F606W" in filters
        assert "F814W" in filters
        assert len(filters) > 10

    def test_calculate_pixel_scale(self, adapter):
        """Test HST pixel scale calculation."""
        # Test ACS/WFC
        data = {"instrument_name": "ACS", "detector": "WFC"}
        assert adapter.calculate_pixel_scale(data) == 0.05

        # Test WFC3/IR
        data = {"instrument_name": "WFC3", "detector": "IR"}
        assert adapter.calculate_pixel_scale(data) == 0.13

        # Test unknown instrument
        data = {"instrument_name": "UNKNOWN"}
        assert adapter.calculate_pixel_scale(data) is None


class TestJWSTAdapter:
    """Test suite for JWST data adapter."""

    @pytest.fixture
    def adapter(self):
        """Create JWST adapter for testing."""
        return JWSTAdapter()

    @pytest.fixture
    def valid_jwst_data(self):
        """Valid JWST observation data."""
        return {
            "obs_id": "jwst_02345_01_nircam_f200w",
            "s_ra": 210.8025,
            "s_dec": 54.3491,
            "t_min": 59500.2,  # MJD
            "filters": "F200W",
            "t_exptime": 1065.0,
            "instrument_name": "NIRCAM",
            "detector": "NRCA1",
            "obs_collection": "JWST",
            "target_name": "NGC5194",
            "proposal_id": "2345",
            "dataURL": "https://mast.stsci.edu/api/v0.1/Download/file?uri=test.fits",
        }

    @pytest.mark.asyncio
    async def test_normalize_observation_data_success(self, adapter, valid_jwst_data):
        """Test successful JWST data normalization."""
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(valid_jwst_data, survey_id)

        assert isinstance(result, ObservationCreate)
        assert result.survey_id == survey_id
        assert result.observation_id == "jwst_02345_01_nircam_f200w"
        assert result.ra == 210.8025
        assert result.dec == 54.3491
        assert result.filter_band == "2.0μm"  # F200W mapped
        assert result.exposure_time == 1065.0
        assert result.airmass is None  # JWST is in space
        assert result.pixel_scale == 0.031  # NIRCam short wavelength

    def test_map_filter_band(self, adapter):
        """Test JWST filter mapping."""
        assert adapter.map_filter_band("F200W") == "2.0μm"
        assert adapter.map_filter_band("F444W") == "4.44μm"
        assert adapter.map_filter_band("UNKNOWN") == "UNKNOWN"

    def test_calculate_pixel_scale(self, adapter):
        """Test JWST pixel scale calculation."""
        # Test NIRCam short wavelength
        data = {"instrument_name": "NIRCAM", "detector": "NRCA1"}
        assert adapter.calculate_pixel_scale(data) == 0.031

        # Test MIRI
        data = {"instrument_name": "MIRI"}
        assert adapter.calculate_pixel_scale(data) == 0.11


class TestSDSSAdapter:
    """Test suite for SDSS data adapter."""

    @pytest.fixture
    def adapter(self):
        """Create SDSS adapter for testing."""
        return SDSSAdapter()

    @pytest.fixture
    def valid_sdss_data(self):
        """Valid SDSS observation data."""
        return {
            "run": 3704,
            "field": 98,
            "camcol": 3,
            "filter": "r",
            "ra": 185.0,
            "dec": 15.8,
            "mjd": 52371.0,
            "exptime": 53.9,
            "objid": 1237654982367920133,
            "clean": 1,
        }

    @pytest.mark.asyncio
    async def test_normalize_observation_data_success(self, adapter, valid_sdss_data):
        """Test successful SDSS data normalization."""
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(valid_sdss_data, survey_id)

        assert isinstance(result, ObservationCreate)
        assert result.survey_id == survey_id
        assert result.observation_id.startswith("sdss-3704-98-3-r")
        assert result.ra == 185.0
        assert result.dec == 15.8
        assert result.filter_band == "r"
        assert result.exposure_time == 53.9
        assert result.pixel_scale == 0.396  # SDSS pixel scale

    @pytest.mark.asyncio
    async def test_normalize_observation_data_with_objid(self, adapter):
        """Test SDSS normalization using objid."""
        data = {
            "objid": 1237654982367920133,
            "ra": 185.0,
            "dec": 15.8,
            "filter": "g",
        }
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(data, survey_id)

        assert result.observation_id == "sdss-1237654982367920133"

    def test_map_filter_band(self, adapter):
        """Test SDSS filter mapping."""
        assert adapter.map_filter_band("u") == "u"
        assert adapter.map_filter_band("g'") == "g"
        assert adapter.map_filter_band("r") == "r"

    def test_extract_image_dimensions(self, adapter):
        """Test SDSS image dimensions."""
        data = {}
        width, height = adapter.extract_image_dimensions(data)
        assert width == 2048
        assert height == 1489


class TestLSSTAdapter:
    """Test suite for LSST data adapter."""

    @pytest.fixture
    def adapter(self):
        """Create LSST adapter for testing."""
        return LSSTAdapter()

    @pytest.fixture
    def valid_lsst_data(self):
        """Valid LSST observation data."""
        return {
            "obsid": 2024030100001,
            "visit": 12345,
            "exposure": 12345001,
            "ra": 150.0,
            "dec": 2.2,
            "filter": "r",
            "exptime": 15.0,
            "mjd": 60000.5,
            "camera": "LSSTCam",
            "detector": "R22_S11",
            "airmass": 1.2,
            "seeing": 0.7,
        }

    @pytest.mark.asyncio
    async def test_normalize_observation_data_success(self, adapter, valid_lsst_data):
        """Test successful LSST data normalization."""
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(valid_lsst_data, survey_id)

        assert isinstance(result, ObservationCreate)
        assert result.survey_id == survey_id
        assert result.observation_id == "lsst-2024030100001"
        assert result.ra == 150.0
        assert result.dec == 2.2
        assert result.filter_band == "r"
        assert result.exposure_time == 15.0
        assert result.airmass == 1.2
        assert result.seeing == 0.7
        assert result.pixel_scale == 0.2  # LSST pixel scale

    @pytest.mark.asyncio
    async def test_normalize_observation_data_with_visit(self, adapter):
        """Test LSST normalization using visit ID."""
        data = {
            "visit": 12345,
            "ra": 150.0,
            "dec": 2.2,
            "filter": "g",
            "detector": "R22_S11",
        }
        survey_id = uuid4()

        result = await adapter.normalize_observation_data(data, survey_id)

        assert result.observation_id == "lsst-12345-R22_S11"

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, adapter, valid_lsst_data):
        """Test LSST metadata extraction."""
        metadata = await adapter.extract_metadata(valid_lsst_data)

        assert metadata["obsid"] == 2024030100001
        assert metadata["visit_id"] == 12345
        assert metadata["camera"] == "LSSTCam"
        assert metadata["detector"] == "R22_S11"
        assert metadata["survey"] == "LSST"
        assert metadata["observatory"] == "Vera C. Rubin Observatory"

    def test_map_filter_band(self, adapter):
        """Test LSST filter mapping."""
        assert adapter.map_filter_band("u") == "u"
        assert adapter.map_filter_band("g") == "g"
        assert adapter.map_filter_band("y") == "y"

    def test_calculate_airmass(self, adapter):
        """Test LSST airmass calculation."""
        # Test direct airmass
        data = {"airmass": 1.5}
        assert adapter.calculate_airmass(data) == 1.5

        # Test calculation from altitude
        data = {"altitude": 60.0}  # 60 degrees
        airmass = adapter.calculate_airmass(data)
        assert airmass is not None
        assert 1.0 < airmass < 2.0

    def test_extract_image_dimensions(self, adapter):
        """Test LSST image dimensions."""
        data = {"camera": "LSSTCam"}
        width, height = adapter.extract_image_dimensions(data)
        assert width == 4096
        assert height == 4096


class TestAdapterCommonFunctionality:
    """Test common functionality across all adapters."""

    @pytest.mark.parametrize(
        "adapter_class", [HSTAdapter, JWSTAdapter, SDSSAdapter, LSSTAdapter]
    )
    def test_get_data_requirements(self, adapter_class):
        """Test that all adapters return data requirements."""
        adapter = adapter_class()
        requirements = adapter.get_data_requirements()

        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "survey_specific" in requirements

    @pytest.mark.parametrize(
        "adapter_class", [HSTAdapter, JWSTAdapter, SDSSAdapter, LSSTAdapter]
    )
    def test_get_supported_filters(self, adapter_class):
        """Test that all adapters return supported filters."""
        adapter = adapter_class()
        filters = adapter.get_supported_filters()

        assert isinstance(filters, list)
        assert len(filters) > 0

    @pytest.mark.parametrize(
        "adapter_class", [HSTAdapter, JWSTAdapter, SDSSAdapter, LSSTAdapter]
    )
    def test_survey_name_initialization(self, adapter_class):
        """Test that adapters are initialized with correct survey names."""
        adapter = adapter_class()

        expected_names = {
            HSTAdapter: "HST",
            JWSTAdapter: "JWST",
            SDSSAdapter: "SDSS",
            LSSTAdapter: "LSST",
        }

        assert adapter.survey_name == expected_names[adapter_class]
