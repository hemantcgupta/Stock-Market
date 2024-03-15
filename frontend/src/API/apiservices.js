import axios from "axios";
import { Component } from "react";
import Swal from "sweetalert2";
import config from "../Constants/config";
import cookieStorage from "../Constants/cookie-storage";
import { string } from "../Constants/string";
import { default as Constant, default as String } from "../Constants/validator";
import api from "./api";

const API_URL = config.API_URL;

const currentYear = new Date().getFullYear();
const currentMonth = new Date().getMonth() + 1;

const DashboardParams = (startDate, endDate, regionId, countryId, cityId) => {
  return `startdate=${startDate}&enddate=${endDate}&regionId=${String.addQuotesforMultiSelect(
    regionId
  )}&countryId=${String.addQuotesforMultiSelect(
    countryId
  )}&cityId=${String.addQuotesforMultiSelect(cityId)}`;
};

const RoouteDashboardParams = (
  startDate,
  endDate,
  regionId,
  countryId,
  routeId
) => {
  return `startdate=${startDate}&enddate=${endDate}&selectedRouteRegion=${String.addQuotesforMultiSelect(
    regionId
  )}&selectedRouteCountry=${String.addQuotesforMultiSelect(
    countryId
  )}&selectedRoute=${String.addQuotesforMultiSelect(routeId)}`;
};

const DemographyDashboardParams = (startDate, endDate, regionId, countryId) => {
  let region =
    typeof regionId === "string"
      ? regionId === "Null"
        ? regionId
        : `${decodeURIComponent(regionId)}`
      : `'${decodeURIComponent(regionId.join("','"))}'`;
  let country =
    typeof countryId === "string"
      ? countryId === "*"
        ? countryId
        : `${countryId}`
      : `'${countryId.join("','")}'`;
  return `startdate=${startDate}&enddate=${endDate}&regionId=${String.addQuotesforMultiSelect(
    region
  )}&countryId=${String.addQuotesforMultiSelect(country)}`;
};

const FilterParams = (regionId, countryId, cityId, getCabinValue) => {
  let cabinValue =
    getCabinValue !== undefined
      ? typeof getCabinValue === "string"
        ? getCabinValue
        : `'${encodeURIComponent(getCabinValue.join("','"))}'`
      : "Null";
  return `regionId=${String.addQuotesforMultiSelect(
    regionId
  )}&countryId=${String.addQuotesforMultiSelect(
    countryId
  )}&cityId=${String.addQuotesforMultiSelect(
    cityId
  )}&getCabinValue=${cabinValue}`;
};

const Params = (regionId, countryId, cityId, getCabinValue) => {
  let region =
    typeof regionId === "string"
      ? regionId === "*"
        ? regionId
        : `${encodeURIComponent(regionId)}`
      : `'${encodeURIComponent(regionId.join("','"))}'`;
  let country =
    typeof countryId === "string"
      ? countryId === "*"
        ? countryId
        : `${countryId}`
      : `'${countryId.join("','")}'`;
  let city =
    typeof cityId === "string"
      ? cityId === "*"
        ? cityId
        : `${cityId}`
      : `'${cityId.join("','")}'`;
  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `regionId=${region}&countryId=${country}&cityId=${city}&getCabinValue=${cabinValue}`;
};

const DemographyParams = (regionId, countryId, getCabinValue) => {
  let region =
    typeof regionId === "string"
      ? regionId === "*"
        ? regionId
        : `${encodeURIComponent(regionId)}`
      : `'${encodeURIComponent(regionId.join("','"))}'`;
  let country =
    typeof countryId === "string"
      ? countryId === "*"
        ? countryId
        : `${encodeURIComponent(countryId)}`
      : `'${encodeURIComponent(countryId.join("','"))}'`;
  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `regionId=${region}&countryId=${country}&getCabinValue=${cabinValue}`;
};

const PromotionParams = (
  regionId,
  countryId,
  serviceGroupId,
  promoTypeId,
  promoTitleId,
  agencyGroupId,
  agentsId,
  commonODId,
  getCabinValue
) => {
  let region =
    typeof regionId === "string"
      ? regionId === "*"
        ? regionId
        : `${encodeURIComponent(regionId)}`
      : `'${encodeURIComponent(regionId.join("','"))}'`;
  let country =
    typeof countryId === "string"
      ? countryId === "*"
        ? countryId
        : `${countryId}`
      : `'${countryId.join("','")}'`;
  let serviceGroup =
    typeof serviceGroupId === "string"
      ? serviceGroupId === "*"
        ? serviceGroupId
        : `${serviceGroupId}`
      : `'${serviceGroupId.join("','")}'`;
  let promoType =
    typeof promoTypeId === "string"
      ? promoTypeId === "*"
        ? promoTypeId
        : `${promoTypeId}`
      : `'${promoTypeId.join("','")}'`;
  let promoTitle =
    typeof promoTitleId === "string"
      ? promoTitleId === "*"
        ? promoTitleId
        : `${promoTitleId}`
      : `'${promoTitleId.join("','")}'`;
  let agencyGroup =
    typeof agencyGroupId === "string"
      ? agencyGroupId === "*"
        ? agencyGroupId
        : `${encodeURIComponent(agencyGroupId)}`
      : `'${encodeURIComponent(agencyGroupId.join("','"))}'`;
  let agents =
    typeof agentsId === "string"
      ? agentsId === "*"
        ? agentsId
        : `${encodeURIComponent(agentsId)}`
      : `'${encodeURIComponent(agentsId.join("','"))}'`;
  let commonOD =
    typeof commonODId === "string"
      ? commonODId === "*"
        ? commonODId
        : `${commonODId}`
      : `'${commonODId.join("','")}'`;
  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `regionId=${region}&countryId=${country}&serviceGroupId=${serviceGroup}&promoTypeId=${promoType}&promotionTitleId=${promoTitle}&agencyGroupId=${agencyGroup}&agentsId=${agents}&commonOD=${commonOD}&getCabinValue=${cabinValue}`;
};

const ROUTEParams = (regionId, countryId, routeId, getCabinValue) => {
  let region =
    typeof regionId === "string"
      ? regionId === "*"
        ? regionId
        : `${encodeURIComponent(regionId)}`
      : `'${encodeURIComponent(regionId.join("','"))}'`;
  let country =
    typeof countryId === "string"
      ? countryId === "*"
        ? countryId
        : `${countryId}`
      : `'${countryId.join("','")}'`;
  let route =
    typeof routeId === "string"
      ? routeId === "*"
        ? routeId
        : `${routeId}`
      : `'${routeId.join("','")}'`;
  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `selectedRouteRegion=${region}&selectedRouteCountry=${country}&selectedRoute=${route}&getCabinValue=${cabinValue}`;
};

const AlertDetailsParams = (
  selectedRouteGroup,
  selectedRegion,
  selectedCountry,
  selectedCity,
  selectedRoute,
  getCabinValue
) => {
  let region =
    typeof selectedRegion === "string"
      ? selectedRegion === "*"
        ? selectedRegion
        : `${encodeURIComponent(selectedRegion)}`
      : `'${encodeURIComponent(selectedRegion.join("','"))}'`;
  let country =
    typeof selectedCountry === "string"
      ? selectedCountry === "*"
        ? selectedCountry
        : `${selectedCountry}`
      : `'${selectedCountry.join("','")}'`;
  let city =
    typeof selectedCity === "string"
      ? selectedCity === "*"
        ? selectedCity
        : `${selectedCity}`
      : `'${selectedCity.join("','")}'`;
  let route =
    typeof selectedRoute === "string"
      ? selectedRoute === "*"
        ? selectedRoute
        : `${selectedRoute}`
      : `'${selectedRoute.join("','")}'`;

  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `selectedRouteGroup=Network&selectedRegion=${region}&selectedCountry=${country}&selectedCity=${city}&selectedRoute=${route}&getCabinValue=${cabinValue}`;
}

const AlertParams = (
  selectedRouteGroup,
  selectedRegion,
  selectedCountry,
  selectedCity,
  selectedRoute,
  getCabinValue
) => {
  let region =
    typeof selectedRegion === "string"
      ? selectedRegion === "*"
        ? selectedRegion
        : `${encodeURIComponent(selectedRegion)}`
      : `'${encodeURIComponent(selectedRegion.join("','"))}'`;
  let country =
    typeof selectedCountry === "string"
      ? selectedCountry === "*"
        ? selectedCountry
        : `${encodeURIComponent(selectedCountry)}`
      : `'${encodeURIComponent(selectedCountry.join("','"))}'`;
  let city =
    typeof selectedCity === "string"
      ? selectedCity === "*"
        ? selectedCity
        : `${encodeURIComponent(selectedCity)}`
      : `'${encodeURIComponent(selectedCity.join("','"))}'`;

  let route =
    typeof selectedRoute === "string"
      ? selectedRoute === "*"
        ? selectedRoute
        : `${encodeURIComponent(selectedRoute)}`
      : `'${encodeURIComponent(selectedRoute.join("','"))}'`;
  console.log("Route", selectedRoute, route);
  let cabinValue =
    typeof getCabinValue === "string"
      ? getCabinValue
      : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

  return `selectedRouteGroup=Network&selectedRegion=${region}&selectedCountry=${country}&selectedCity=${city}&selectedRoute=${route}&getCabinValue=${cabinValue}`;
};


// const width = window.innerWidth;
// const  = width < 720 ? width / 5 : width / 18
// const  = width < 720 ? width / 5 : width / 15

export default class APIServices extends Component {
  constructor(props) {
    super(props);
    this.state = {
      alertVisible: false,
    };
  }

  Params = (regionId, countryId, cityId, getCabinValue) => {
    let region =
      typeof regionId === "string"
        ? regionId === "*"
          ? regionId
          : `${encodeURIComponent(regionId)}`
        : `'${encodeURIComponent(regionId.join("','"))}'`;
    let country =
      typeof countryId === "string"
        ? countryId === "*"
          ? countryId
          : `${countryId}`
        : `'${countryId.join("','")}'`;
    let city =
      typeof cityId === "string"
        ? cityId === "*"
          ? cityId
          : `${cityId}`
        : `'${cityId.join("','")}'`;
    let cabinValue =
      typeof getCabinValue === "string"
        ? getCabinValue
        : `'${encodeURIComponent(getCabinValue.join("','"))}'`;

    return `regionId=${region}&countryId=${country}&cityId=${city}&getCabinValue=${cabinValue}`;
  };

  //Constant Functions
  getDefaultHeader = () => {
    const token = cookieStorage.getCookie("Authorization");
    return {
      headers: {
        Authorization: token,
      },
    };
  };

  errorHandling = (error) => {
    console.log("error::::::", error, error.response);
    if (error && error.response) {
      if (error.response.status === 500) {
        Swal.fire({
          title: "Error!",
          text: "Something went wrong. Please try after some time",
          icon: "error",
          confirmButtonText: "Ok",
        });
      } else if (error.response.status === 403) {
        Swal.fire({
          title: "Error!",
          text: "Authorization failed! (Your token has been expired. Please login again)",
          icon: "error",
          confirmButtonText: "Ok",
        }).then(() => {
          window.location = "/";
        });
      }
    }
  };

  statusArrowIndicator = (params) => {
    var element = document.createElement("span");
    var icon = document.createElement("i");

    // visually indicate if this months value is higher or lower than last months value
    let value = params.value;
    if (value === "Above") {
      icon.className = "fa fa-arrow-up";
    } else if (value === "Below") {
      icon.className = "fa fa-arrow-down";
    } else {
      icon.className = "";
    }
    element.appendChild(document.createTextNode(value));
    element.appendChild(icon);
    return element;

  }

  arrowIndicator = (params) => {
    var element = document.createElement("span");
    var icon = document.createElement("i");

    // visually indicate if this months value is higher or lower than last months value
    let VLY = params.value ? params.value : "0";
    const numericVLY = parseFloat(VLY);
    if (VLY === "---") {
      icon.className = "";
    } else if (typeof VLY === "string") {
      if (VLY.includes("B") || VLY.includes("M") || VLY.includes("K")) {
        icon.className = "fa fa-arrow-up";
      } else if (numericVLY > 0) {
        icon.className = "fa fa-arrow-up";
      } else if (numericVLY < 0) {
        icon.className = "fa fa-arrow-down";
      } else {
        icon.className = "";
      }
    } else if (numericVLY > 0) {
      icon.className = "fa fa-arrow-up";
    } else if (numericVLY < 0) {
      icon.className = "fa fa-arrow-down";
    } else {
      icon.className = "";
    }
    element.appendChild(document.createTextNode(VLY));
    element.appendChild(icon);
    return element;
  };

  costArrowIndicator = (params) => {
    var element = document.createElement("span");
    var icon = document.createElement("i");

    // visually indicate if this months value is higher or lower than last months value
    let VLY = params.value ? params.value : "0";
    const numericVLY = parseFloat(VLY);
    if (VLY === "---") {
      icon.className = "";
    } else if (typeof VLY === "string") {
      if (VLY.includes("B") || VLY.includes("M") || VLY.includes("K")) {
        icon.className = "fa cost-up fa-arrow-up";
      } else if (numericVLY > 0) {
        icon.className = "fa cost-up fa-arrow-up";
      } else if (numericVLY < 0) {
        icon.className = "fa cost-down fa-arrow-down";
      } else {
        icon.className = "";
      }
    } else if (numericVLY > 0) {
      icon.className = "fa cost-up fa-arrow-up";
    } else if (numericVLY < 0) {
      icon.className = "fa cost-down fa-arrow-down";
    } else {
      icon.className = "";
    }
    element.appendChild(document.createTextNode(VLY));
    element.appendChild(icon);
    return element;
  };

  accuracyArrowIndicator = (params) => {
    var element = document.createElement("span");
    var icon = document.createElement("i");
    let VLY = params.value ? params.value : "0";
    const accuracy = parseFloat(VLY);
    if (accuracy === 0) {
      icon.className = "";
    } else if (accuracy >= 86 && accuracy <= 100) {
      icon.className = "fa fa-arrow-up";
    } else if (accuracy <= 86 || accuracy >= 100) {
      icon.className = "fa fa-arrow-down";
    } else {
      icon.className = "";
    }
    element.appendChild(document.createTextNode(VLY));
    element.appendChild(icon);
    return element;
  };

  barsIndicator() {
    var element = document.createElement("span");
    var icon = document.createElement("i");
    icon.className = "fa fa-bar-chart-o";
    element.appendChild(icon);
    return element;
  }

  convertZeroValueToBlank(Value) {
    let convertedValue =
      window.numberFormat(Value) === 0 ? "0" : window.numberFormat(Value);
    return convertedValue;
  }

  topperFixed(Value) {
    let convertedValue =
      window.numberFormat(Value, 2) === "0.00"
        ? "---"
        : window.numberFormat(Value, 2);
    return convertedValue;
  }

  showPercent(value) {
    return value === null || value === 0 ? "" : "%";
  }

  customSorting = (a, b) => {
    let valueA = window.convertNumberFormat(a);
    let valueB = window.convertNumberFormat(b);
    return valueA - valueB;
  };

  numberSorting = (a, b) => {
    return a - b;
  };

  getUserPreferences(key, value, count) {
    let userData = cookieStorage.getCookie("userDetails");
    userData = userData ? JSON.parse(userData) : "";
    const userPreference = api
      .get(`userpreferences`, "hideloader")
      .then((response) => {
        if (response && response.data.response.length > 0) {
          const responseData = response.data.response[0];
          if (responseData[key] !== value) {
            this.updateUserPreferences(key, value, userData);
            const valueWithoutNull = responseData[key] ? responseData[key] : 0;
            return count ? value - valueWithoutNull : true;
          } else {
            return count ? 0 : false;
          }
        } else {
          this.postUserPreferences(key, value, userData);
          return count ? 1 : true;
        }
      })
      .catch((err) => {
        console.log("user preferences error", err);
      });

    return userPreference;
  }

  postUserPreferences(key, value, userData) {
    const postData = {
      user_email: userData.email,
      [key]: value,
    };
    api
      .post(`userpreferences`, postData, "hideloader")
      .then((response) => {
        console.log("user preferences response", response);
      })
      .catch((err) => {
        console.log("user preferences error", err);
      });
  }

  updateUserPreferences(key, value, userData) {
    const updatedData = {
      user_email: userData.email,
      [key]: value,
    };
    api
      .put(`userpreferences`, updatedData, "hideloader")
      .then((response) => {
        console.log("user preferences response", response);
      })
      .catch((err) => {
        console.log("user preferences error", err);
      });
  }

  //Dashboard Common API
  getMonthsRange() {
    const url = `${API_URL}/getmonthrange`;
    var monthsRange = axios
      .get(url)
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
    return monthsRange;
  }

  getRegions() {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/getRegion`;
    var regions = axios
      .get(url)
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
    return regions;
  }

  getCountries(regionId) {
    let region =
      typeof regionId === "string"
        ? regionId
        : `'${encodeURIComponent(regionId.join("','"))}'`;
    const url = `${API_URL}/getCountryByRegionId?regionId=${region}`;
    var countries = axios
      .get(url)
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
    return countries;
  }

  getCities(regionId, countryId) {
    let country =
      typeof countryId === "string" ? countryId : `'${countryId.join("','")}'`;
    let region =
      typeof regionId === "string" ? regionId : `'${regionId.join("','")}'`;
    const url = `${API_URL}/getCityByCountryCode?regionId=${region}&countryCode=${country}`;
    var cities = axios
      .get(url)
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
    return cities;
  }

  getODs(regionId, countryId, cityId, searchod) {
    let country =
      typeof countryId === "string" ? countryId : `'${countryId.join("','")}'`;
    let region =
      typeof regionId === "string" ? regionId : `'${regionId.join("','")}'`;
    let city = typeof cityId === "string" ? cityId : `'${cityId.join("','")}'`;
    const url = `${API_URL}/getCommonODByCity?regionId=${region}&countryCode=${country}&cityCode=${city}&searchOD=${searchod}`;
    var ODs = axios
      .get(url)
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
    return ODs;
  }

  getIndicatorsData(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    dashboard
  ) {
    const header = this.getDefaultHeader();
    let link = "";
    let params = null;
    if (dashboard === "Pos") {
      link = "posytd";
      params = DashboardParams(startDate, endDate, regionId, countryId, cityId);
    } else if (dashboard === "Route") {
      link = "routeytd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    } else if (dashboard === "Route Revenue Planning") {
      link = "routeytd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    } else if (dashboard === "Route Profitability") {
      link = "rpmtd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    } else if (dashboard === "Demography") {
      link = "demographicYTD";
      params = DemographyDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId
      );
    }
    const url = `${API_URL}/${link}?${params}`;
    var indicatorsData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return indicatorsData;
  }

  getAnciallaryItems(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    posDashboard
  ) {
    let that = this;
    const link = posDashboard ? `posancillarytable` : `routeancillarytable`;
    const params = posDashboard
      ? DashboardParams(startDate, endDate, regionId, countryId, cityId)
      : `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    const url = `${API_URL}/${link}?${params}`;
    var ancillaryData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.ANCIALLARY_ITEMS,
            headerTooltip: "Anciallary",
            field: "Anciallary",
            tooltipField: "Anciallary",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: that.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: that.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VTG,
            headerTooltip: "VTG(%)",
            field: "VTG",
            tooltipField: "VTG_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var ancillaryDetails = [];
        response.data.response[0].TableData.forEach(function (key) {
          ancillaryDetails.push({
            Anciallary: key.AncillaryName,
            AnciallaryCodeName: key.AncillaryCode,
            CY: that.convertZeroValueToBlank(key.Ancillary_CY),
            LY: that.convertZeroValueToBlank(key.Ancillary_LY),
            VLY: that.convertZeroValueToBlank(key.Ancillary_VLY),
            VTG: that.convertZeroValueToBlank(key.Ancillary_VTG),
            CY_AB: window.numberWithCommas(key.Ancillary_CY),
            LY_AB: window.numberWithCommas(key.Ancillary_LY),
            VLY_AB: window.numberWithCommas(key.Ancillary_VLY),
            VTG_AB: window.numberWithCommas(key.Ancillary_VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Anciallary: "Total",
            AnciallaryCodeName: key.AncillaryCode,
            CY: that.convertZeroValueToBlank(key.Ancillary_CY),
            LY: that.convertZeroValueToBlank(key.Ancillary_LY),
            VLY: that.convertZeroValueToBlank(key.Ancillary_VLY),
            VTG: that.convertZeroValueToBlank(key.Ancillary_VTG),
            CY_AB: window.numberWithCommas(key.Ancillary_CY),
            LY_AB: window.numberWithCommas(key.Ancillary_LY),
            VLY_AB: window.numberWithCommas(key.Ancillary_VLY),
            VTG_AB: window.numberWithCommas(key.Ancillary_VTG),
          });
        });

        return [
          {
            columnName: columnName,
            ancillaryDetails: ancillaryDetails,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ancillaryData;
  }

  getCabinBudget(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    posDashboard
  ) {
    let that = this;
    const link = posDashboard ? `posancillarytable` : `routeancillarytable`;
    const params = posDashboard
      ? DashboardParams(startDate, endDate, regionId, countryId, cityId)
      : `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    const url = `${API_URL}/${link}?${params}`;
    var ancillaryData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.Cabin,
            headerTooltip: "Cabin",
            field: "Cabin",
            tooltipField: "Cabin",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.BUDGET,
            headerTooltip: "Budget",
            field: "Budget",
            tooltipField: "Budget_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var ancillaryDetails = [];
        response.data.response[0].TableData.forEach(function (key) {
          ancillaryDetails.push({
            Cabin: key.AncillaryName,
            Budget: that.convertZeroValueToBlank(key.Ancillary_CY),
            LY: that.convertZeroValueToBlank(key.Ancillary_LY),
            VLY: that.convertZeroValueToBlank(key.Ancillary_VLY),
            Budget_AB: window.numberWithCommas(key.Ancillary_CY),
            LY_AB: window.numberWithCommas(key.Ancillary_LY),
            VLY_AB: window.numberWithCommas(key.Ancillary_VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Cabin: "Total",
            Budget: that.convertZeroValueToBlank(key.Ancillary_CY),
            LY: that.convertZeroValueToBlank(key.Ancillary_LY),
            VLY: that.convertZeroValueToBlank(key.Ancillary_VLY),
            Budget_AB: window.numberWithCommas(key.Ancillary_CY),
            LY_AB: window.numberWithCommas(key.Ancillary_LY),
            VLY_AB: window.numberWithCommas(key.Ancillary_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            ancillaryDetails: ancillaryDetails,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ancillaryData;
  }

  getDataLoadIndicator() {
    const url = `${API_URL}/dataloadedindicator`;
    var dataloadindicator = axios
      .get(url)
      .then((response) => {
        var columnName = [
          {
            headerName: "Sr.No.",
            headerTooltip: "Sr.No.",
            field: "Sr_No",
            tooltipField: "Sr_No",
            alignLeft: true,
            width: 100,
          },
          {
            headerName: "Data Source",
            headerTooltip: "Data Source",
            field: "DataSource",
            tooltipField: "DataSource",
            alignLeft: true,
          },
          {
            headerName: "Refresh Frequency",
            headerTooltip: "Refresh Frequency",
            field: "Refresh_Frequency",
            tooltipField: "Refresh_Frequency",
            alignLeft: true,
          },
          {
            headerName: "Last Updated Date",
            headerTooltip: "Last Updated Date",
            field: "Last_Updated_Date",
            tooltipField: "Last_Updated_Date",
            alignLeft: true,
            width: 100,
          },
          {
            headerName: "Status",
            headerTooltip: "Status",
            field: "Status",
            tooltipField: "Status",
            alignLeft: true,
            width: 100,
            cellStyle: (params) => {
              var Status = params.data.Status;
              let color = "";
              if (Status) {
                if (Status.toLowerCase().includes("s")) {
                  color = "rgb(17, 247, 17)";
                } else if (Status.toLowerCase().includes("f")) {
                  color = "rgb(216, 62, 62)";
                }
              }
              return {
                color: color,
                "text-transform": "capitalize",
              };
            },
          },
          {
            headerName: "Comment",
            headerTooltip: "Comment",
            field: "Comment",
            tooltipField: "Comment",
            alignLeft: true,
          },
        ];

        var rowData = [];
        response.data.response.forEach(function (key, i) {
          rowData.push({
            Sr_No: i + 1,
            DataSource: key.DataSource,
            Refresh_Frequency: key.Refresh_Frequency,
            Last_Updated_Date: key.Last_Updated_Date,
            Status: key.Status,
            Comment: key.Comments,
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return dataloadindicator;
  }

  //POS Dashboard API
  getODsBarChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/postop5ODsmultibarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getAgentsBarChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/postop10agentsmultibarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRegionBarChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/posperformanceview?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getSegmentationBarChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/possegmentationbarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getSalesAnalysisLineChart(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    trend,
    graphType,
    selectedGraph
  ) {
    const url = `${API_URL}/possalesanalysis?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&trend=${trend}&graphType=${graphType}&graphCategory=${selectedGraph}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getPOSCards(startDate, endDate, regionId, countryId, cityId) {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/poscards?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var cardData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return cardData;
  }

  getChanneltable(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/poschannelperformanceview?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var route = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.CHANNEL,
            headerTooltip: "Channel",
            field: "Channel",
            tooltipField: "Channel",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VTG,
            headerTooltip: "VTG(%)",
            field: "VTG",
            tooltipField: "VTG_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
            sortable: true,
            comparator: this.customSorting,
          },
          // { headerName: 'Budget', headerTooltip: 'Budget', field: 'Budget', tooltipField: 'Budget', width: 250 }
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].TableData.forEach(function (key) {
          rowData.push({
            Channel: key.ChannelName,
            CY: that.convertZeroValueToBlank(key.Revenue_CY),
            LY: that.convertZeroValueToBlank(key.Revenue_LY),
            VLY: that.convertZeroValueToBlank(key.Revenue_VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.Revenue_CY),
            LY_AB: window.numberWithCommas(key.Revenue_LY),
            VLY_AB: window.numberWithCommas(key.Revenue_VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Channel: "Total",
            CY: that.convertZeroValueToBlank(key.Revenue_CY),
            LY: that.convertZeroValueToBlank(key.Revenue_LY),
            VLY: that.convertZeroValueToBlank(key.Revenue_VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.Revenue_CY),
            LY_AB: window.numberWithCommas(key.Revenue_LY),
            VLY_AB: window.numberWithCommas(key.Revenue_VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          { columnName: columnName, rowData: rowData, totalData: totalData },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return route;
  }

  getMarketShare(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/posmarketsharetable?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var marketShareDatas = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.OD,
            headerTooltip: "OD",
            field: "OD",
            tooltipField: "OD",
            width: 200,
            alignLeft: true,
          },
          {
            headerName: "CMSV",
            headerTooltip: "CMSV",
            field: "CMSV",
            tooltipField: "CMSV_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY_CM",
            tooltipField: "VLY_CM_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: "CALMS",
            headerTooltip: "CALMS",
            field: "CALMS",
            tooltipField: "CALMS_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY_CA",
            tooltipField: "VLY_CA_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response.forEach(function (key) {
          rowData.push({
            OD: key.OD,
            CMSV: that.convertZeroValueToBlank(key.CMSV_CY),
            VLY_CM: that.convertZeroValueToBlank(key.CMSV_VLY),
            CALMS: that.convertZeroValueToBlank(key.CALMS_CY),
            VLY_CA: that.convertZeroValueToBlank(key.CALMS_VLY),
            CMSV_AB: window.numberWithCommas(key.CMSV_CY),
            VLY_CM_AB: window.numberWithCommas(key.CMSV_VLY),
            CALMS_AB: window.numberWithCommas(key.CALMS_CY),
            VLY_CA_AB: window.numberWithCommas(key.CALMS_VLY),
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return marketShareDatas;
  }

  getDDSChartData(startDate, endDate, regionId, countryId, cityId, dataType) {
    const url = `${API_URL}/ddschart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&dataType=${dataType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  //Interline Page API

  getInterlineMonthTables(
    currency,
    partnersId,
    regionId,
    countryId,
    cityId,
    segmentId,
    channelId,
    getCabinValue,
    oneWorldValue
  ) {
    const url = `${API_URL}/InterlineMonthly?partnerId=${String.addQuotesforMultiSelect(
      partnersId
    )}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&segmentsId=${String.addQuotesforMultiSelect(
      segmentId
    )}&channelId=${channelId}&oneWorld=${String.addQuotesforMultiSelect(
      oneWorldValue
    )}`;

    var posmonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RR",
                tooltipField: "CY_RR_AB",
                underline: true,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RR",
                tooltipField: "VLY_RR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RG",
                tooltipField: "CY_RG_AB",
                width: 250,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RG",
                tooltipField: "VLY_RG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.YQ_RETAINED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Y",
                tooltipField: "CY_Y_AB",
                width: 250,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Y",
                tooltipField: "VLY_Y_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.PAX_SECTOR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                width: 250,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CR",
                tooltipField: "CY_CR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CR",
                tooltipField: "VLY_CR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CG",
                tooltipField: "CY_CG_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CG",
                tooltipField: "VLY_CG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.MH_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.PARTNERSHIP_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_PR",
                tooltipField: "CY_PR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_PR",
                tooltipField: "VLY_PR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        let previosYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear - 1
        );
        let currentYearTableDta = response.data.TableData.filter(
          (d) => d.Year === currentYear
        );
        let nextYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear + 1
        );

        var responseData = [
          ...response.data.Total_LY,
          ...previosYearTableData,
          ...currentYearTableDta,
          ...response.data.Total_NY,
          ...nextYearTableData,
        ];

        var rowData = [];

        responseData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),
            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        var totalData = [];
        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),
            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return posmonthtable;
  }

  getInterlineDrillDownData(
    getYear,
    currency,
    gettingMonth,
    partnersId,
    regionId,
    countryId,
    cityId,
    segmentId,
    channelId,
    getCabinValue,
    oneWorldValue,
    type
  ) {
    // let selectedType = type;
    // if (commonOD !== '*') {
    //     if (type == 'Ancillary' || type == 'Agency') {
    //         selectedType = 'Null';
    //     }
    // }

    const url = `${API_URL}/InterlineDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&partnerId=${String.addQuotesforMultiSelect(
      partnersId
    )}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&segmentsId=${String.addQuotesforMultiSelect(
      segmentId
    )}&channelId=${channelId}&oneWorld=${String.addQuotesforMultiSelect(
      oneWorldValue
    )}&type=${type}`;

    // const downloadUrl = `${API_URL}/FullYearDownloadPOS?getYear=${getYear}&${Params(regionId, countryId, cityId, getCabinValue)}&commonOD=${encodeURIComponent(commonOD)}&type=${selectedType}`;
    // console.log(getYear, 'latest')
    // localStorage.setItem('postype', type)
    // localStorage.setItem('posDownloadURL', downloadUrl)

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;

        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  (type === "Null" ||
                    type === "Agency" ||
                    type === "Channel") &&
                    firstColumnName !== "Cabin" &&
                    firstColumnName !== "Agent"
                    ? true
                    : false,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: '', field: '', cellRenderer: (params) => this.alerts(params), width: 150 }]
          // },
          {
            headerName: string.columnName.REVENUE_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RR",
                tooltipField: "CY_RR_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RR",
                tooltipField: "VLY_RR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RG",
                tooltipField: "CY_RG_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RG",
                tooltipField: "VLY_RG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.YQ_RETAINED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Y",
                tooltipField: "CY_Y_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Y",
                tooltipField: "VLY_Y_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.PAX_SECTOR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CR",
                tooltipField: "CY_CR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CR",
                tooltipField: "VLY_CR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CG",
                tooltipField: "CY_CG_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CG",
                tooltipField: "VLY_CG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.MH_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.PARTNERSHIP_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_PR",
                tooltipField: "CY_PR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_PR",
                tooltipField: "VLY_PR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            "": "",
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),
            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
            isAlert: key.is_alert,
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),
            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
            isAlert: key.is_alert,
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return posregiontable;
  }

  getInterlineDetails(
    getYear,
    gettingMonth,
    partnersId,
    regionId,
    countryId,
    cityId,
    segmentId,
    channelId,
    getCabinValue,
    oneWorldValue
  ) {
    const url = `${API_URL}/InterlineTarget?getYear=${getYear}&gettingMonth=${gettingMonth}`;

    var cabinTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        console.log(response, "response");
        var columnName = [
          // {
          //     headerName: 'Month', field: 'Month', tooltipField: 'Month_AB', alignLeft: true
          // },
          {
            headerName: "CY",
            field: "CY",
            tooltipField: "CY_AB",
          },
          {
            headerName: "TGT",
            field: "TGT(%)",
            tooltipField: "TGT(%)_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
          // {
          //     headerName: 'Ticketed Average Fare(SR)', field: 'Ticketed Average Fare(SR)', tooltipField: 'Ticketed Average Fare(SR)_AB'
          // },
          {
            headerName: "VTG(%)",
            field: "VTG(%)TKT",
            tooltipField: "VTG(%)TKT_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
        ];

        // var F = response.data.Data.filter((d) => d.Cabin === 'F')
        // var J = response.data.Data.filter((d) => d.Cabin === 'J')
        // var Y = response.data.Data.filter((d) => d.Cabin === 'Y')

        // var Total_F = response.data.Total.filter((d) => d.RBD === 'Total of F')
        // var Total_J = response.data.Total.filter((d) => d.RBD === 'Total of J')
        // var Total_Y = response.data.Total.filter((d) => d.RBD === 'Total of Y')

        //var mergedCabinData = [...response]
        var cabinData = [];
        cabinData.push({
          CY: this.convertZeroValueToBlank(
            response.data.Data[0].RevenueReceived_CY
          ),
          "TGT(%)": this.convertZeroValueToBlank(
            response.data.Data[0].RevenueReceived_TGT
          ),
          "VTG(%)TKT": this.convertZeroValueToBlank(
            response.data.Data[0].RevenueReceived_VTG
          ),
          CY_AB: window.numberWithCommas(
            response.data.Data[0].RevenueReceived_CY
          ),
          "TGT(%)_AB": window.numberWithCommas(
            response.data.Data[0].RevenueReceived_TGT
          ),
          "VTG(%)TKT_AB": window.numberWithCommas(
            response.data.Data[0].RevenueReceived_VTG
          ),
        });
        // });
        console.log(cabinData, "cabin");
        return [
          {
            columnName: columnName,
            cabinData: cabinData,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return cabinTable;
  }

  getInterlineChannelDetails(
    currency,
    getYear,
    gettingMonth,
    partnersId,
    regionId,
    countryId,
    cityId,
    segmentId,
    channelId,
    getCabinValue,
    oneWorldValue,
    type,
    count
  ) {
    const url = `${API_URL}/InterlineAgencyData?getYear=${getYear}&gettingMonth=${gettingMonth}&partnerId=${String.addQuotesforMultiSelect(
      partnersId
    )}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&segmentsId=${String.addQuotesforMultiSelect(
      segmentId
    )}&channelId=${channelId}&oneWorld=${String.addQuotesforMultiSelect(
      oneWorldValue
    )}&type=${type}&pageNum=${count}`;

    // const downloadUrl = `${API_URL}/FullYearDownloadPOS?getYear=${getYear}&${Params(regionId, countryId, cityId, getCabinValue)}&commonOD=${encodeURIComponent(commonOD)}&type=${selectedType}`;
    // console.log(getYear, 'latest')
    // localStorage.setItem('postype', type)
    // localStorage.setItem('posDownloadURL', downloadUrl)

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        // const firstColumnName = response.data.ColumnName;

        // let avgfarezeroTGT = response.data.TableData.filter((d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null)
        // let avgfareTGTVisible = avgfarezeroTGT.length === response.data.TableData.length

        // let revenuzeroTGT = response.data.TableData.filter((d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null)
        // let revenueTGTVisible = revenuzeroTGT.length === response.data.TableData.length

        // let passengerzeroTGT = response.data.TableData.filter((d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null)
        // let passengerTGTVisible = passengerzeroTGT.length === response.data.TableData.length

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: "Agent",
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: '', field: '', cellRenderer: (params) => this.alerts(params), width: 150 }]
          // },
          {
            headerName: string.columnName.REVENUE_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RR",
                tooltipField: "CY_RR_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RR",
                tooltipField: "VLY_RR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_RG",
                tooltipField: "CY_RG_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_RG",
                tooltipField: "VLY_RG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.YQ_RETAINED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Y",
                tooltipField: "CY_Y_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Y",
                tooltipField: "VLY_Y_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.PAX_SECTOR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_RECEIVED,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CR",
                tooltipField: "CY_CR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CR",
                tooltipField: "VLY_CR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.COMMISION_GIVEN,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_CG",
                tooltipField: "CY_CG_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_CG",
                tooltipField: "VLY_CG_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.MH_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.PARTNERSHIP_AVG_FARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_PR",
                tooltipField: "CY_PR_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_PR",
                tooltipField: "VLY_PR_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.Agent === null ? "---" : key.Agent,
            "": "",
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),

            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
            isAlert: key.is_alert,
          });
        });
        console.log(rowData, "row");

        var totalData = [];
        response.data.response.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            CY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_CY)
                : this.convertZeroValueToBlank(key.RevenueReceived_CY),
            VLY_RR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueReceived_VLY)
                : this.convertZeroValueToBlank(key.RevenueReceived_VLY),
            CY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_CY)
                : this.convertZeroValueToBlank(key.RevenueGiven_CY),
            VLY_RG:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenueGiven_VLY)
                : this.convertZeroValueToBlank(key.RevenueGiven_VLY),
            CY_Y: this.convertZeroValueToBlank(key.YQRetained_CY),
            VLY_Y: this.convertZeroValueToBlank(key.YQRetained_VLY),
            CY_P: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_P: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_CR: this.convertZeroValueToBlank(key.CommisionReceived_CY),
            VLY_CR: this.convertZeroValueToBlank(key.CommisionReceived_VLY),
            CY_CG: this.convertZeroValueToBlank(key.CommisionGiven_CY),
            VLY_CG: this.convertZeroValueToBlank(key.CommisionGiven_VLY),
            CY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_CY)
                : this.convertZeroValueToBlank(key.MHAvgFare_CY),
            VLY_M:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalMHAvgFare_VLY)
                : this.convertZeroValueToBlank(key.MHAvgFare_VLY),
            CY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_CY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_CY),
            VLY_PR:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalPartnershipAvgFare_VLY)
                : this.convertZeroValueToBlank(key.PartnershipAvgFare_VLY),

            CY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_CY)
                : window.numberWithCommas(key.RevenueReceived_CY),
            VLY_RR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueReceived_VLY)
                : window.numberWithCommas(key.RevenueReceived_VLY),
            CY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_CY)
                : window.numberWithCommas(key.RevenueGiven_CY),
            VLY_RG_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenueGiven_VLY)
                : window.numberWithCommas(key.RevenueGiven_VLY),
            CY_Y_AB: window.numberWithCommas(key.YQRetained_CY),
            VLY_Y_AB: window.numberWithCommas(key.YQRetained_VLY),
            CY_P_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_P_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_CR_AB: window.numberWithCommas(key.CommisionReceived_CY),
            VLY_CR_AB: window.numberWithCommas(key.CommisionReceived_VLY),
            CY_CG_AB: window.numberWithCommas(key.CommisionGiven_CY),
            VLY_CG_AB: window.numberWithCommas(key.CommisionGiven_VLY),
            CY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_CY)
                : window.numberWithCommas(key.MHAvgFare_CY),
            VLY_M_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalMHAvgFare_VLY)
                : window.numberWithCommas(key.MHAvgFare_VLY),
            CY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_CY)
                : window.numberWithCommas(key.PartnershipAvgFare_CY),
            VLY_PR_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalPartnershipAvgFare_VLY)
                : window.numberWithCommas(key.PartnershipAvgFare_VLY),
            isAlert: key.is_alert,
          });
        });
        console.log(rowData, "total");

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentPage: response.data.response.pageNumber,
            totalPages: response.data.response.totalPages,
            totalRecords: response.data.response.totalRecords,
            paginationSize: response.data.response.paginationLimit,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return posregiontable;
  }

  //Route Dashboard API
  getRouteRegions(routeGroup) {
    let routeGroups =
      typeof routeGroup === "string"
        ? routeGroup
        : `'${encodeURIComponent(routeGroup.join("','"))}'`;
    const url = `${API_URL}/getRouteRegion?routeGroup=${routeGroups}`;
    var regions = axios
      .get(url)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return regions;
  }

  getRouteCountries(routeGroup, regionId) {
    let routeGroups =
      typeof routeGroup === "string"
        ? routeGroup
        : `'${encodeURIComponent(routeGroup.join("','"))}'`;
    let region =
      typeof regionId === "string"
        ? regionId
        : `'${encodeURIComponent(regionId.join("','"))}'`;
    const url = `${API_URL}/getRouteCountryByRegionId?routeGroup=${routeGroups}&regionId=${region}`;
    var countries = axios
      .get(url)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return countries;
  }

  getRoutes(routeGroup, regionId, countryId) {
    let routeGroups =
      typeof routeGroup === "string"
        ? routeGroup
        : `'${encodeURIComponent(routeGroup.join("','"))}'`;
    let region =
      typeof regionId === "string" ? regionId : `'${regionId.join("','")}'`;
    let country =
      typeof countryId === "string" ? countryId : `'${countryId.join("','")}'`;
    const url = `${API_URL}/getRouteCityByCountryCode?routeGroup=${routeGroups}&regionId=${region}&countryId=${country}`;
    var routes = axios
      .get(url)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return routes;
  }

  getRevenueBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    graphType
  ) {
    const url = `${API_URL}/routerevenuemultibarchartv2?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRaskBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    graphType
  ) {
    const url = `${API_URL}/routeraskmultibarchartv2?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getYeildBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    graphType
  ) {
    const url = `${API_URL}/routeyieldmultibarchartv2?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getLoadFactorBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    graphType
  ) {
    const url = `${API_URL}/routeloadfactormultibarchartv2?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRegionWisePerformanceTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId
  ) {
    const url = `${API_URL}/routeregionwisetable?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}`;
    var Regionwiseperformance = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var tableHead = response.data.response[0].ColumnName;
        var columnName = [
          {
            headerName: tableHead,
            headerTooltip: tableHead,
            field: "ColumnName",
            tooltipField: "Region",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VTG,
            headerTooltip: "VTG(%)",
            field: "VTG",
            tooltipField: "VTG_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].Data.forEach(function (key) {
          rowData.push({
            ColumnName: key.ColumnName,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            ColumnName: "Total",
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            tableHead: tableHead,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return Regionwiseperformance;
  }

  getRouteTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    toporbottom
  ) {
    const url = `${API_URL}/routetop5routestable?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&toporbottom=${toporbottom}`;
    var route = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.ROUTE,
            headerTooltip: string.columnName.ROUTE,
            field: "Route",
            tooltipField: "Route",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: string.columnName.CY,
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: string.columnName.LY,
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: string.columnName.VLY,
            field: "VLY",
            tooltipField: "VLY_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VTG,
            headerTooltip: string.columnName.VTG,
            field: "VTG",
            tooltipField: "VTG_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response.forEach(function (key) {
          rowData.push({
            Route: key.Routes,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return route;
  }

  getRouteCards(startDate, endDate, routeGroup, regionId, countryId, routeId) {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/routecards?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}`;
    var cardData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return cardData;
  }

  // Route Profitability Dashboard
  getSurplusDeficitBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend
  ) {
    const url = `${API_URL}/rpsurplusdeficit?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getBreakevenLoadFactorAvgBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    dropdown
  ) {
    const url = `${API_URL}/rploadfactororavgfare?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphDropdown=${dropdown}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getCaskRaskBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    dropdown
  ) {
    const url = `${API_URL}/rpcaskorrask?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphDropdown=${dropdown}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getForexFuelBarChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend,
    dropdown
  ) {
    const url = `${API_URL}/rpforexorfuel?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}&graphDropdown=${dropdown}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRouteProfitabilityCards(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId
  ) {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/rpcards?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}`;
    var cardData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return cardData;
  }

  getAircraftPerformance(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    typeofCost
  ) {
    let that = this;
    const params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&viewBy=${typeofCost}`;
    const url = `${API_URL}/rpaircraftperformance?${params}`;

    const isCost = typeofCost.includes("C") ? true : false;

    var ancillaryData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.Aircraft_Type,
            headerTooltip: "Aircraft_Type",
            field: "Aircraft_Type",
            tooltipField: "Aircraft_Type",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: "VBGT",
            headerTooltip: "VBGT",
            field: "VTG",
            tooltipField: "VTG_AB",
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var ancillaryDetails = [];
        response.data.response[0].TableData.forEach(function (key) {
          ancillaryDetails.push({
            Aircraft_Type: key.Aircraft,
            CY: that.convertZeroValueToBlank(key.CY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            VLY: that.convertZeroValueToBlank(key.VLY),
            CY_AB: window.numberWithCommas(key.CY),
            VTG_AB: window.numberWithCommas(key.VTG),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Aircraft_Type: "Total",
            CY: that.convertZeroValueToBlank(key.CY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            VLY: that.convertZeroValueToBlank(key.VLY),
            CY_AB: window.numberWithCommas(key.CY),
            VTG_AB: window.numberWithCommas(key.VTG),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        return [
          {
            columnName: columnName,
            ancillaryDetails: ancillaryDetails,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ancillaryData;
  }

  getRouteRegionPerformanceTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    typeofCost
  ) {
    const url = `${API_URL}/rpregionperformance?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&viewBy=${typeofCost}`;

    const isCost = typeofCost.includes("C") ? true : false;

    var Regionwiseperformance = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var tableHead = response.data.response[0].ColumnName;
        var columnName = [
          {
            headerName: tableHead,
            headerTooltip: tableHead,
            field: "ColumnName",
            tooltipField: "ColumnName",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: "VBGT",
            headerTooltip: "VBGT",
            field: "VTG",
            tooltipField: "VTG_AB",
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].Data.forEach(function (key) {
          rowData.push({
            ColumnName: key.ColumnName,
            CY: that.convertZeroValueToBlank(key.CY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            VLY: that.convertZeroValueToBlank(key.VLY),
            CY_AB: window.numberWithCommas(key.CY),
            VTG_AB: window.numberWithCommas(key.VTG),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            ColumnName: "Total",
            CY: that.convertZeroValueToBlank(key.CY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            VLY: that.convertZeroValueToBlank(key.VLY),
            CY_AB: window.numberWithCommas(key.CY),
            VTG_AB: window.numberWithCommas(key.VTG),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            tableHead: tableHead,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return Regionwiseperformance;
  }

  getTopTenRouteBudgetTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    typeofCost
  ) {
    const url = `${API_URL}/rptop10routes?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&viewBy=${typeofCost}`;

    const isCost = typeofCost.includes("C") ? true : false;

    var route = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.ROUTE,
            headerTooltip: string.columnName.ROUTE,
            field: "Route",
            tooltipField: "Route",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: "VBGT",
            headerTooltip: "VBGT",
            field: "VTG",
            tooltipField: "VTG_AB",
            cellRenderer: isCost
              ? this.costArrowIndicator
              : this.arrowIndicator,
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response.forEach(function (key) {
          rowData.push({
            Route: key.Route,
            CY: that.convertZeroValueToBlank(key.CY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            VLY: that.convertZeroValueToBlank(key.VLY),
            CY_AB: window.numberWithCommas(key.CY),
            VTG_AB: window.numberWithCommas(key.VTG),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return route;
  }

  //Route Revenue Planning Dashboard
  getRevenueBudgetChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend
  ) {
    const url = `${API_URL}/routerevenuemultibarchart?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRaskBudgetChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend
  ) {
    const url = `${API_URL}/routeraskmultibarchart?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getYeildBudgetChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend
  ) {
    const url = `${API_URL}/routeyieldmultibarchart?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getLoadFactorBudgetChart(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    trend
  ) {
    const url = `${API_URL}/routeloadfactormultibarchart?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&trend=${trend}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRegionBudgetTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId
  ) {
    const url = `${API_URL}/routeregionwisetable?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}`;
    var Regionwiseperformance = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var tableHead = response.data.response[0].ColumnName;
        var columnName = [
          {
            headerName: tableHead,
            headerTooltip: tableHead,
            field: "ColumnName",
            tooltipField: "ColumnName",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.BUDGET,
            headerTooltip: "Budget",
            field: "Budget",
            tooltipField: "Budget_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].Data.forEach(function (key) {
          rowData.push({
            ColumnName: key.ColumnName,
            Budget: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VTG),
            Budget_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            ColumnName: "Total",
            Budget: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VTG),
            Budget_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            tableHead: tableHead,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return Regionwiseperformance;
  }

  getRouteBudgetTable(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId,
    toporbottom
  ) {
    const url = `${API_URL}/routetop5routestable?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}&toporbottom=${toporbottom}`;
    var route = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.ROUTE,
            headerTooltip: string.columnName.ROUTE,
            field: "Route",
            tooltipField: "Route",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.BUDGET,
            headerTooltip: "Budget",
            field: "Budget",
            tooltipField: "Budget_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response.forEach(function (key) {
          rowData.push({
            Route: key.Routes,
            Budget: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            Budget_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return route;
  }

  getRouteBudgetCards(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    routeId
  ) {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/routecards?selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      routeId
    )}`;
    var cardData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return cardData;
  }

  //POS Revenue Planning Dashboard
  getODsBudgetChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/postop5ODsmultibarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getAgentsBudgetChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/postop10agentsmultibarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRegionBudgetChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/posperformanceview?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getTargetRationaltChart(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/possegmentationbarchart?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getSalesAnalysisBudgetChart(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    trend,
    graphType,
    selectedGraph
  ) {
    const url = `${API_URL}/possalesanalysis?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&trend=${trend}&graphType=${graphType}&graphCategory=${selectedGraph}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getChannelBudgetTable(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/poschannelperformanceview?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var channelBudget = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.CHANNEL,
            headerTooltip: "Channel",
            field: "Channel",
            tooltipField: "Channel",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.BUDGET,
            headerTooltip: "Budget",
            field: "Budget",
            tooltipField: "Budget_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY",
            tooltipField: "VLY_AB",
            width: 250,
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].TableData.forEach(function (key) {
          rowData.push({
            Channel: key.ChannelName,
            Budget: that.convertZeroValueToBlank(key.Revenue_CY),
            LY: that.convertZeroValueToBlank(key.Revenue_LY),
            VLY: that.convertZeroValueToBlank(key.VTG),
            Budget_AB: window.numberWithCommas(key.Revenue_CY),
            LY_AB: window.numberWithCommas(key.Revenue_LY),
            VLY_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Channel: "Total",
            Budget: that.convertZeroValueToBlank(key.Revenue_CY),
            LY: that.convertZeroValueToBlank(key.Revenue_LY),
            VLY: that.convertZeroValueToBlank(key.VTG),
            Budget_AB: window.numberWithCommas(key.Revenue_CY),
            LY_AB: window.numberWithCommas(key.Revenue_LY),
            VLY_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          { columnName: columnName, rowData: rowData, totalData: totalData },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return channelBudget;
  }

  getODBudgetTable(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/posmarketsharetable?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var ODBudgetDatas = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.OD,
            headerTooltip: "OD",
            field: "OD",
            tooltipField: "OD",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: "BGT",
            headerTooltip: "BGT",
            field: "BGT",
            tooltipField: "BGT_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.VLY,
            headerTooltip: "VLY(%)",
            field: "VLY_CA",
            tooltipField: "VLY_CA_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
          {
            headerName: "CALMS",
            headerTooltip: "CALMS",
            field: "CALMS",
            tooltipField: "CALMS_AB",
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response.forEach(function (key) {
          rowData.push({
            OD: key.OD,
            BGT: that.convertZeroValueToBlank(key.CMSV_CY),
            CALMS: that.convertZeroValueToBlank(key.CALMS_CY),
            VLY_CA: that.convertZeroValueToBlank(key.CALMS_VLY),
            BGT_AB: window.numberWithCommas(key.CMSV_CY),
            CALMS_AB: window.numberWithCommas(key.CALMS_CY),
            VLY_CA_AB: window.numberWithCommas(key.CALMS_VLY),
          });
        });

        return [{ columnName: columnName, rowData: rowData }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ODBudgetDatas;
  }

  getPRPIndicatorsData(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    dashboard
  ) {
    const header = this.getDefaultHeader();
    let link = "";
    let params = null;
    if (dashboard === "Pos") {
      link = "posytd";
      params = DashboardParams(startDate, endDate, regionId, countryId, cityId);
    } else if (dashboard === "Route") {
      link = "routeytd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    } else if (dashboard === "Route Revenue Planning") {
      link = "routeytd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    } else if (dashboard === "Route Profitability") {
      link = "routeytd";
      params = `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    }
    const url = `${API_URL}/${link}?${params}`;
    var indicatorsData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return indicatorsData;
  }

  //Agent Dashboard API
  getIncrementalRO(
    gettingYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    currency
  ) {
    const url = `${API_URL}/revenueopportunity?getYear=${gettingYear}&gettingMonth=${gettingMonth}&${FilterParams(
      regionId,
      countryId,
      cityId
    )}&currencyType=${currency}`;
    var ROData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.CHANNEL,
                field: "Channel",
                alignLeft: true,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.RO,
                field: "RO",
                tooltipField: "RO_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
            ],
          },
          {
            headerName: string.columnName.SALES,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Sales",
                tooltipField: "CY_Sales_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_Sales",
                tooltipField: "YOY_Sales_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_B",
                tooltipField: "YOY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MS",
                tooltipField: "CY_MS_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_MS",
                tooltipField: "YOY_MS_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response[0].TableData.forEach(function (key) {
          rowData.push({
            Channel: key.ChannelName,
            RO:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.RO_BaseCurrency)
                : that.convertZeroValueToBlank(key.RO_LocalCurrency),
            CY_Sales:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.CY_Revenue_BaseCurrency)
                : that.convertZeroValueToBlank(key.CY_Revenue_LocalCurrency),
            YOY_Sales:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.VLY_Revenue_BaseCurrency)
                : that.convertZeroValueToBlank(key.VLY_Revenue_LocalCurrency),
            CY_B: that.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: that.convertZeroValueToBlank(key.Bookings_LY),
            YOY_B: that.convertZeroValueToBlank(key.Bookings_YOY),
            CY_MS: that.convertZeroValueToBlank(key.Share_CY),
            YOY_MS: that.convertZeroValueToBlank(key.Share_YOY),
            RO_AB:
              currency === "BC"
                ? window.numberWithCommas(key.RO_BaseCurrency)
                : window.numberWithCommas(key.RO_LocalCurrency),
            CY_Sales_AB:
              currency === "BC"
                ? window.numberWithCommas(key.CY_Revenue_BaseCurrency)
                : window.numberWithCommas(key.CY_Revenue_LocalCurrency),
            YOY_Sales_AB:
              currency === "BC"
                ? window.numberWithCommas(key.VLY_Revenue_BaseCurrency)
                : window.numberWithCommas(key.VLY_Revenue_LocalCurrency),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            LY_B_AB: window.numberWithCommas(key.Bookings_LY),
            YOY_B_AB: window.numberWithCommas(key.Bookings_YOY),
            CY_MS_AB: window.numberWithCommas(key.Share_CY),
            YOY_MS_AB: window.numberWithCommas(key.Share_YOY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Channel: "Total",
            RO:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.RO_BaseCurrency)
                : that.convertZeroValueToBlank(key.RO_LocalCurrency),
            CY_Sales:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.CY_Revenue_BaseCurrency)
                : that.convertZeroValueToBlank(key.CY_Revenue_LocalCurrency),
            YOY_Sales:
              currency === "BC"
                ? that.convertZeroValueToBlank(key.VLY_Revenue_BaseCurrency)
                : that.convertZeroValueToBlank(key.VLY_Revenue_LocalCurrency),
            CY_B: that.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: that.convertZeroValueToBlank(key.Bookings_LY),
            YOY_B: that.convertZeroValueToBlank(key.Bookings_YOY),
            CY_MS: that.convertZeroValueToBlank(key.Share_CY),
            YOY_MS: that.convertZeroValueToBlank(key.Share_YOY),
            RO_AB:
              currency === "BC"
                ? window.numberWithCommas(key.RO_BaseCurrency)
                : window.numberWithCommas(key.RO_LocalCurrency),
            CY_Sales_AB:
              currency === "BC"
                ? window.numberWithCommas(key.CY_Revenue_BaseCurrency)
                : window.numberWithCommas(key.CY_Revenue_LocalCurrency),
            YOY_Sales_AB:
              currency === "BC"
                ? window.numberWithCommas(key.VLY_Revenue_BaseCurrency)
                : window.numberWithCommas(key.VLY_Revenue_LocalCurrency),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            LY_B_AB: window.numberWithCommas(key.Bookings_LY),
            YOY_B_AB: window.numberWithCommas(key.Bookings_YOY),
            CY_MS_AB: window.numberWithCommas(key.Share_CY),
            YOY_MS_AB: window.numberWithCommas(key.Share_YOY),
          });
        });

        return [
          { columnName: columnName, rowData: rowData, totalData: totalData },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ROData;
  }

  getTopAgentWithOD(
    gettingYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    currency
  ) {
    const url = `${API_URL}/topagentandtopOD?${FilterParams(
      regionId,
      countryId,
      cityId
    )}&getYear=${gettingYear}&gettingMonth=${gettingMonth}&currencyType=${currency}`;
    var TopAgentWithOD = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                field: "OD",
                alignLeft: true,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.RO,
                field: "RO",
                tooltipField: "RO_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
            ],
          },
          {
            headerName: string.columnName.SALES,
            children: [
              {
                headerName: string.columnName.REVENUE,
                field: "Revenue",
                tooltipField: "Revenue_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_Rev",
                tooltipField: "YOY_Rev_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.PASSENGER,
                field: "Pax",
                tooltipField: "Pax_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_Pax",
                tooltipField: "YOY_Pax_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.AVERAGE_FARE_$,
                field: "AvgFare",
                tooltipField: "AvgFare_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_Avg",
                tooltipField: "YOY_Avg_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.FIT,
                field: "FIT",
                tooltipField: "FIT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_FIT",
                tooltipField: "YOY_FIT_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.GIT,
                field: "GIT",
                tooltipField: "GIT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_GIT",
                tooltipField: "YOY_GIT_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.SHARE,
                field: "Share",
                tooltipField: "Share_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_Share",
                tooltipField: "YOY_Share_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              // { headerName: string.columnName.FARE_SHARE, field: 'Fare_Share' },
              // { headerName: string.columnName.TOP_AL_SHARE, field: 'Top_AL_Share' },
            ],
          },
        ];

        let data = response.data.response;
        console.log("rahul", data);
        data.forEach((data) => {
          data.Data.forEach((key) => {
            return (
              (key["OD"] = key.OD),
              (key["RO"] =
                currency === "BC"
                  ? this.convertZeroValueToBlank(key.RO_BaseCurrency)
                  : this.convertZeroValueToBlank(key.RO_LocalCurrency)),
              (key["Revenue"] =
                currency === "BC"
                  ? this.convertZeroValueToBlank(key.CY_Revenue_BaseCurrency)
                  : this.convertZeroValueToBlank(key.CY_Revenue_LocalCurrency)),
              (key["YOY_Rev"] =
                currency === "BC"
                  ? this.convertZeroValueToBlank(key.VLY_Revenue_BaseCurrency)
                  : this.convertZeroValueToBlank(
                    key.VLY_Revenue_LocalCurrency
                  )),
              (key["Pax"] = this.convertZeroValueToBlank(
                key.TotalPassenger_CY
              )),
              (key["YOY_Pax"] = this.convertZeroValueToBlank(key.YOY_Pax)),
              (key["AvgFare"] =
                currency === "BC"
                  ? this.convertZeroValueToBlank(key.BaseCurrency_AvgFare_CY)
                  : this.convertZeroValueToBlank(key.LocalCurrency_AvgFare_CY)),
              (key["YOY_Avg"] =
                currency === "BC"
                  ? this.convertZeroValueToBlank(key.VLY_BaseCurrency_Avg)
                  : this.convertZeroValueToBlank(key.VLY_LocalCurrency_Avg)),
              (key["FIT"] = this.convertZeroValueToBlank(key.FIT)),
              (key["YOY_FIT"] = this.convertZeroValueToBlank(key.YOY_FIT)),
              (key["GIT"] = this.convertZeroValueToBlank(key.GIT)),
              (key["YOY_GIT"] = this.convertZeroValueToBlank(key.YOY_GIT)),
              (key["Share"] = this.convertZeroValueToBlank(key.Share)),
              (key["YOY_Share"] = this.convertZeroValueToBlank(key.YOY_Share)),
              (key["RO_AB"] =
                currency === "BC"
                  ? window.numberWithCommas(key.RO_BaseCurrency)
                  : window.numberWithCommas(key.RO_LocalCurrency)),
              (key["Revenue_AB"] =
                currency === "BC"
                  ? window.numberWithCommas(key.CY_Revenue_BaseCurrency)
                  : window.numberWithCommas(key.CY_Revenue_LocalCurrency)),
              (key["YOY_Rev_AB"] =
                currency === "BC"
                  ? window.numberWithCommas(key.VLY_Revenue_BaseCurrency)
                  : window.numberWithCommas(key.VLY_Revenue_LocalCurrency)),
              (key["Pax_AB"] = window.numberWithCommas(key.TotalPassenger_CY)),
              (key["YOY_Pax_AB"] = window.numberWithCommas(key.YOY_Pax)),
              (key["AvgFare_AB"] =
                currency === "BC"
                  ? window.numberWithCommas(key.BaseCurrency_AvgFare_CY)
                  : window.numberWithCommas(key.LocalCurrency_AvgFare_CY)),
              (key["YOY_Avg_AB"] =
                currency === "BC"
                  ? window.numberWithCommas(key.VLY_BaseCurrency_Avg)
                  : window.numberWithCommas(key.VLY_LocalCurrency_Avg)),
              (key["FIT_AB"] = window.numberWithCommas(key.FIT)),
              (key["YOY_FIT_AB"] = window.numberWithCommas(key.YOY_FIT)),
              (key["GIT_AB"] = window.numberWithCommas(key.GIT)),
              (key["YOY_GIT_AB"] = window.numberWithCommas(key.YOY_GIT)),
              (key["Share_AB"] = window.numberWithCommas(key.Share)),
              (key["YOY_Share_AB"] = window.numberWithCommas(key.YOY_Share))
              // key['Fare_Share'] = '87',
              // key['Top_AL_Share'] = 'BA/60%'
            );
          });
        });
        console.log("rahul", data);
        return [{ columnName: columnName, rowData: data }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log("rahul error", error);
      });

    return TopAgentWithOD;
  }

  //Others
  getRealTimeRevenueMultilineChart(getXaxis) {
    const url = `${API_URL}/realtimerevenueMultiline?Xaxis=${getXaxis}`;

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });

    return posregiontable;
  }

  getInFareMultilineChart(regionId, countryId, cityId, time) {
    const url = `${API_URL}/infaremultiline?regionId=${regionId}&countryId=${countryId}&cityId=${cityId}&time=${time}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data)
      .catch((error) => {
        // this.errorHandling(error);
      });
  }

  getRegionWisePerformance(regionId, countryId, cityId, commonOD) {
    const url = `${API_URL}/performanceview?regionId=${regionId}&countryId=${countryId}&cityId=${cityId}&commonOD=${commonOD}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
  }

  gettop5ODsBarChart(gettingMonth, regionId, countryId, cityId, commonOD) {
    const url = `${API_URL}/top5ODsMultiBarchart?gettingMonth=${gettingMonth}&regionId=${regionId}&countryId=${countryId}&cityId=${cityId}&commonOD=${commonOD}`;

    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data)
      .catch((error) => {
        console.log(error);
      });
  }

  getChannelWisePerformPivot(regionId, countryId, cityId) {
    const url = `${API_URL}/channelperformanceview?regionId=${regionId}&countryId=${countryId}&cityId=${cityId}`;
    var channelwiseperform = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.CHANNEL,
            field: "Channel",
            width: 300,
          },
          { headerName: string.columnName.CY, field: "CY" },
          { headerName: string.columnName.LY, field: "LY" },
          {
            headerName: string.columnName.VLY,
            field: "VLY",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ];

        var channelwiseperformDatas = [];
        let that = this;
        response.data.forEach(function (key) {
          channelwiseperformDatas.push({
            Channel: key.ChannelName,
            CY: that.convertZeroValueToBlank(key.Revenue_CY),
            LY: that.convertZeroValueToBlank(key.Revenue_LY),
            VLY: that.convertZeroValueToBlank(key.Revenue_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            channelwiseperformtableDatas: channelwiseperformDatas,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return channelwiseperform;
  }

  // Demography Dashboard API
  getDemographyTop10Routes(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/top20InternDomRouteSearch?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyTop20LocationVisitors(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId
  ) {
    const url = `${API_URL}/top20GeoGraphicLocationVisitor?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyUniqueVisitors(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/UniqueVisitorBoughtDropout?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyAgeGroupData(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/AgeGroupGraph?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyTop10Routes(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/top20InternDomRouteSearch?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyTop20LocationVisitors(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId
  ) {
    const url = `${API_URL}/top20GeoGraphicLocationVisitor?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyUniqueVisitors(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/UniqueVisitorBoughtDropout?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemographyAgeGroupData(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    graphType
  ) {
    const url = `${API_URL}/AgeGroupGraph?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}&graphType=${graphType}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getDemoBottomCards(startDate, endDate, regionId, countryId, cityId) {
    const header = this.getDefaultHeader();
    const url = `${API_URL}/demographicMTD?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var cardData = axios
      .get(url, header)
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
    return cardData;
  }

  getEnrichTiertable(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/enrichTable?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var route = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "EnrichTier",
            headerTooltip: "EnrichTier",
            field: "EnrichTier",
            tooltipField: "EnrichTier",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: this.customSorting,
          },
          // { headerName: string.columnName.VLY, headerTooltip: 'VLY(%)', field: "VLY", tooltipField: 'VLY_AB', cellRenderer: (params) => this.arrowIndicator(params), width: 250, sortable: true, comparator: this.customSorting },
          // { headerName: string.columnName.VTG, headerTooltip: 'VTG(%)', field: "VTG", tooltipField: 'VTG_AB', cellRenderer: (params) => this.arrowIndicator(params), width: 250, sortable: true, comparator: this.customSorting },
          // { headerName: 'Budget', headerTooltip: 'Budget', field: 'Budget', tooltipField: 'Budget', width: 250 }
        ];

        let that = this;
        var rowData = [];
        response.data.response[0].Data.forEach(function (key) {
          rowData.push({
            EnrichTier: key.Enrich_tier,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            EnrichTier: "Total",
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          { columnName: columnName, rowData: rowData, totalData: totalData },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return route;
  }

  getNationalityTableItems(
    startDate,
    endDate,
    routeGroup,
    regionId,
    countryId,
    cityId,
    routeId,
    posDashboard
  ) {
    let that = this;
    const params = posDashboard
      ? DashboardParams(startDate, endDate, regionId, countryId, cityId)
      : `selectedRouteGroup=${routeGroup}&${RoouteDashboardParams(
        startDate,
        endDate,
        regionId,
        countryId,
        routeId
      )}`;
    const url = `${API_URL}/nationalityTable?${params}`;
    var ancillaryData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "Nationality",
            headerTooltip: "Nationality",
            field: "Nationality",
            tooltipField: "Nationality",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: string.columnName.CY,
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: that.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            sortable: true,
            comparator: that.customSorting,
          },
        ];

        var ancillaryDetails = [];
        response.data.response[0].Data.forEach(function (key) {
          ancillaryDetails.push({
            Nationality: key.nationality,
            NationalityCodeName: key.AncillaryCode,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Nationality: "Total",
            NationalityCodeName: key.AncillaryCode,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            VLY: that.convertZeroValueToBlank(key.VLY),
            VTG: that.convertZeroValueToBlank(key.VTG),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
            VLY_AB: window.numberWithCommas(key.VLY),
            VTG_AB: window.numberWithCommas(key.VTG),
          });
        });

        return [
          {
            columnName: columnName,
            ancillaryDetails: ancillaryDetails,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return ancillaryData;
  }

  getFlightPurchaseTable(startDate, endDate, regionId, countryId, cityId) {
    const url = `${API_URL}/AFPTable?${DashboardParams(
      startDate,
      endDate,
      regionId,
      countryId,
      cityId
    )}`;
    var marketShareDatas = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "Range",
            headerTooltip: "Range",
            field: "Range",
            tooltipField: "Range",
            width: 200,
            alignLeft: true,
          },
          {
            headerName: "CY",
            headerTooltip: "CY",
            field: "CY",
            tooltipField: "CY_AB",
            sortable: true,
            comparator: this.customSorting,
            sort: "desc",
          },
          {
            headerName: string.columnName.LY,
            headerTooltip: "LY",
            field: "LY",
            tooltipField: "LY_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: true,
            comparator: this.customSorting,
          },
        ];

        var rowData = [];
        let that = this;
        response.data.response[0].Data.forEach(function (key) {
          rowData.push({
            Range: key.AFP_Range,
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
          });
        });
        var totalData = [];
        response.data.response[0].Total.forEach(function (key) {
          totalData.push({
            Range: "Total",
            CY: that.convertZeroValueToBlank(key.CY),
            LY: that.convertZeroValueToBlank(key.LY),
            CY_AB: window.numberWithCommas(key.CY),
            LY_AB: window.numberWithCommas(key.LY),
          });
        });

        return [
          { columnName: columnName, rowData: rowData, totalData: totalData },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return marketShareDatas;
  }

  //Demography Report Page Api

  getDemographyMonthTables(
    currency,
    regionId,
    countryId,
    commonOD,
    cabinId,
    customerSegmentationId,
    enrichId,
    getCabinValue
  ) {
    const url = `${API_URL}/DemographyMonthly?${DemographyParams(
      regionId,
      countryId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&cabinId=${String.addQuotesforMultiSelect(
      cabinId
    )}&customerSegment=${customerSegmentationId}&enrich=${enrichId}`;

    var posmonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: "Pax(Sector)",
            // headerGroupComponent: 'customHeaderGroupComponent',
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_S",
                tooltipField: "CY_S_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_S",
                tooltipField: "VLY_S_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: "Pax(Unique)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_U",
                tooltipField: "CY_U_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_U",
                tooltipField: "VLY_U_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },

          {
            headerName: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        let previosYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear - 1
        );
        let currentYearTableDta = response.data.TableData.filter(
          (d) => d.Year === currentYear
        );
        let nextYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear + 1
        );

        var responseData = [
          ...response.data.Total_LY,
          ...previosYearTableData,
          ...currentYearTableDta,
          ...response.data.Total_NY,
          ...nextYearTableData,
        ];

        var rowData = [];

        responseData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            CY_S: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_S: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_U: this.convertZeroValueToBlank(key.PaxUnique_CY),
            VLY_U: this.convertZeroValueToBlank(key.PaxUnique_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_S_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_S_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_U_AB: window.numberWithCommas(key.PaxUnique_CY),
            VLY_U_AB: window.numberWithCommas(key.PaxUnique_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        console.log(rowData, columnName, "kmkm");
        var totalData = [];
        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_S: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_S: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_U: this.convertZeroValueToBlank(key.PaxUnique_CY),
            VLY_U: this.convertZeroValueToBlank(key.PaxUnique_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_S_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_S_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_U_AB: window.numberWithCommas(key.PaxUnique_CY),
            VLY_U_AB: window.numberWithCommas(key.PaxUnique_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return posmonthtable;
  }

  getDemographyDrillDownData(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    commonOD,
    cabinId,
    customerSegmentationId,
    enrichId,
    getCabinValue,
    type
  ) {
    let selectedType = type;
    if (commonOD !== "*") {
      if (type == "Ancillary" || type == "Agency") {
        selectedType = "Null";
      }
    }
    // const downloadUrl = `${API_URL}/FullYearDownloadPOS?getYear=${getYear}&${Params(regionId, countryId, getCabinValue)}&commonOD=${encodeURIComponent(commonOD)}&cabin=${cabinId}&enrich=${enrichId}&type=${selectedType}`;

    // localStorage.setItem('postype', type)
    // localStorage.setItem('posDownloadURL', downloadUrl)

    const url = `${API_URL}/DemographyDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&${DemographyParams(
      regionId,
      countryId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&cabinId=${String.addQuotesforMultiSelect(
      cabinId
    )}&customerSegmentId=${String.addQuotesforMultiSelect(
      customerSegmentationId
    )}&enrich=${enrichId}&type=${type}`;

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;

        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  (type === "Null" ||
                    type === "Nationality" ||
                    type === "Age Band" ||
                    type === "Customer Segmentation") &&
                    firstColumnName !== "Cabin Category"
                    ? true
                    : false,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: '', field: '', cellRenderer: (params) => this.alerts(params), width: 150 }]
          // },
          {
            headerName: "Pax(Sector)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_S",
                tooltipField: "CY_S_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_S",
                tooltipField: "VLY_S_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "Pax(Unique)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_U",
                tooltipField: "CY_U_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_U",
                tooltipField: "VLY_U_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },

          {
            headerName: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            "": "",
            CY_S: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_S: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_U: this.convertZeroValueToBlank(key.PaxUnique_CY),
            VLY_U: this.convertZeroValueToBlank(key.PaxUnique_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_S_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_S_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_U_AB: window.numberWithCommas(key.PaxUnique_CY),
            VLY_U_AB: window.numberWithCommas(key.PaxUnique_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            isAlert: key.is_alert,
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            CY_S: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_S: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_U: this.convertZeroValueToBlank(key.PaxUnique_CY),
            VLY_U: this.convertZeroValueToBlank(key.PaxUnique_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_S_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_S_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_U_AB: window.numberWithCommas(key.PaxUnique_CY),
            VLY_U_AB: window.numberWithCommas(key.PaxUnique_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            isAlert: key.is_alert,
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return posregiontable;
  }

  getDemographySegmentationDetails(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    commonOD,
    cabinId,
    customerSegmentationId,
    customerSegmentationId_1,
    enrichId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/DemographySegmentationData?getYear=${getYear}&gettingMonth=${gettingMonth}&${DemographyParams(
      regionId,
      countryId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&cabinId=${String.addQuotesforMultiSelect(
      cabinId
    )}&customerSegmentId=${String.addQuotesforMultiSelect(
      customerSegmentationId
    )}&enrich=${enrichId}&customerSegmentId_1=%27${customerSegmentationId_1}%27&type=${type}`;

    var customrSegmentationTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: "Segmentation",
                field: "Seg",
                tooltipField: "Seg_AB",
                width: 300,
                alignLeft: true,
              },
            ],
          },
          {
            headerName: "Pax(Sector)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_S",
                tooltipField: "CY_S_AB",
                //sortable: true, comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_S",
                tooltipField: "VLY_S_AB",
                //sortable: true, comparator: this.customSorting,
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: "Pax(Unique)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_U",
                tooltipField: "CY_U_AB",
                //sortable: true, comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_U",
                tooltipField: "VLY_U_AB",
                //sortable: true, comparator: this.customSorting,
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
                //sortable: true, comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                //sortable: true, comparator: this.customSorting,
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        let uniqueSegments = [];
        let responseData = response.data.response;
        let tableData = responseData.TableData;
        let totalData = responseData.Total;
        let tableDataResponses = [];
        let totalDataResponses = [];

        var keyCount = 0;
        for (var i = 0; i < tableData.length; i++) {
          if (
            uniqueSegments.indexOf(tableData[i]["CustomerSegmentation_2"]) < 0
          ) {
            uniqueSegments.push(tableData[i]["CustomerSegmentation_2"]);
            let tempObj = tableData.filter(
              (d) =>
                d.CustomerSegmentation_2 ===
                tableData[i]["CustomerSegmentation_2"]
            );
            tableDataResponses[keyCount] = tempObj;
            let totalTempObj = totalData.filter(
              (d) =>
                d.CustomerSegmentation_3 ===
                "Total of " + tableData[i]["CustomerSegmentation_2"]
            );
            totalDataResponses[keyCount] = totalTempObj;
            keyCount++;
          }
        }

        var x = 0;
        let mergedDataArr = [];
        for (var j = 0; j < uniqueSegments.length; j++) {
          mergedDataArr.push(totalDataResponses[j][0]);
          for (var k = 0; k < tableDataResponses[j].length; k++) {
            x++;
            mergedDataArr.push(tableDataResponses[j][k]);
          }
        }
        var segmentData = [];

        mergedDataArr.forEach((key) => {
          segmentData.push({
            Segmentation: key.CustomerSegmentation_2,
            Seg: key.CustomerSegmentation_3,
            CY_S: this.convertZeroValueToBlank(key.PaxSector_CY),
            VLY_S: this.convertZeroValueToBlank(key.PaxSector_VLY),
            CY_U: this.convertZeroValueToBlank(key.PaxUnique_CY),
            VLY_U: this.convertZeroValueToBlank(key.PaxUnique_VLY),
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            Seg_AB: key.CustomerSegmentation_3,
            CY_S_AB: window.numberWithCommas(key.PaxSector_CY),
            VLY_S_AB: window.numberWithCommas(key.PaxSector_VLY),
            CY_U_AB: window.numberWithCommas(key.PaxUnique_CY),
            VLY_U_AB: window.numberWithCommas(key.PaxUnique_VLY),
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            segmentData: segmentData,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return customrSegmentationTable;
  }

  getNationalityGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    commonOD,
    cabinId,
    enrichId,
    getCabinValue,
    type,
    NationalityId
  ) {
    const url = `${API_URL}/nationalityAgeBandGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${DemographyParams(
      regionId,
      countryId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&cabinId=${String.addQuotesforMultiSelect(
      cabinId
    )}&enrich=${enrichId}&type=${type}&Nationality=${NationalityId}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getAgeBandGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    commonOD,
    cabinId,
    enrichId,
    getCabinValue,
    type,
    AgeBandId
  ) {
    const url = `${API_URL}/nationalityAgeBandGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${DemographyParams(
      regionId,
      countryId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&cabinId=${String.addQuotesforMultiSelect(
      cabinId
    )}&enrich=${enrichId}&type=${type}&ageBand=${AgeBandId}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  // Demography Top 20 Dom/Int report

  getDemographyReportData(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue
  ) {
    const url = `${API_URL}/demographicReportPage?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}`;

    // const downloadurl = `${API_URL}/FullYearDownloadSegment?endDate=${endDate}&startDate=${startDate}`;

    // localStorage.setItem('segmentationDownloadurl', downloadurl)

    var segmentationreport = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.response.filter(
          (d) => d.AverageFare_TGT === 0 || d.AverageFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        let passengerzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.response.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: "Lead Time",
                field: "Segment",
                tooltipField: "Segment",
                width: 250,
                alignLeft: true,
                underline: false,
              },
            ],
          },
          {
            headerName: "Pax",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                // cellRenderer: (params) => this.arrowIndicator(params), sortable: true, comparator: this.customSorting
              },

              {
                headerName: string.columnName.LY,
                field: "LY_P",
                tooltipField: "LY_P_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "Bought",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CYY_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },

              {
                headerName: string.columnName.LY,
                field: "LY_B",
                tooltipField: "LY_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "DropOut",
            // headerGroupComponent: 'customHeaderGroupComponent',
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_D",
                tooltipField: "CY_D_AB",
                sortable: true,
                comparator: this.customSorting,
              },

              {
                headerName: string.columnName.LY,
                field: "LY_D",
                tooltipField: "LY_D_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          console.log(key, "key");
          rowData.push({
            Segment: key.Lead_Time,
            CY_P: this.convertZeroValueToBlank(key.Pax_CY),
            LY_P: this.convertZeroValueToBlank(key.Pax_LY),
            CY_B: this.convertZeroValueToBlank(key.Bought_CY),
            LY_B: this.convertZeroValueToBlank(key.Bought_LY),
            CY_D: this.convertZeroValueToBlank(key.DropOut_CY),
            LY_D: this.convertZeroValueToBlank(key.DropOut_LY),
            CY_P_AB: window.numberWithCommas(key.Pax_CY),
            LY_P_AB: window.numberWithCommas(key.Pax_LY),
            CY_B_AB: this.convertZeroValueToBlank(key.Bought_CY),
            LY_B_AB: this.convertZeroValueToBlank(key.Bought_LY),
            CY_D_AB: this.convertZeroValueToBlank(key.DropOut_CY),
            LY_D_AB: this.convertZeroValueToBlank(key.DropOut_LY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Segment: "Total",
            CY_P: this.convertZeroValueToBlank(key.Pax_CY),
            LY_P: this.convertZeroValueToBlank(key.Pax_LY),
            CY_B: this.convertZeroValueToBlank(key.Bought_CY),
            LY_B: this.convertZeroValueToBlank(key.Bought_LY),
            CY_D: this.convertZeroValueToBlank(key.DropOut_CY),
            LY_D: this.convertZeroValueToBlank(key.DropOut_LY),
            CY_P_AB: window.numberWithCommas(key.Pax_CY),
            LY_P_AB: window.numberWithCommas(key.Pax_LY),
            CY_B_AB: this.convertZeroValueToBlank(key.Bought_CY),
            LY_B_AB: this.convertZeroValueToBlank(key.Bought_LY),
            CY_D_AB: this.convertZeroValueToBlank(key.DroOout_CY),
            LY_D_AB: this.convertZeroValueToBlank(key.DropOut_LY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return segmentationreport;
  }

  getTopGeographicalDemographyChartData(
    startDate,
    endDate,
    regionId,
    countryId
  ) {
    console.log(regionId, "region");
    const url = `${API_URL}/top20GeoGraphicReportPage?${DemographyDashboardParams(
      startDate,
      endDate,
      regionId,
      countryId
    )}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  // /alertmonthly?selectedRouteGroup=Network&selectedRegion=*&selectedCountry=*&selectedRoute=*&getCabinValue=Null&SelectedCity
  getAlertMonthTables(
    routeGroup,
    region,
    country,
    city,
    route,
    cabinValue,
    alertType
  ) {
    const url = `${API_URL}/alertmonthly?${AlertParams(
      routeGroup,
      region,
      country,
      city,
      route,
      cabinValue,
    )}&alertType=${alertType}`;

    var alertMonthTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let currentYearTotal = response.data.Total_CY

        var responseData = [
          ...response.data.TableData,
        ];

        var rowData = [];

        responseData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        var totalData = [];
        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        const columnName = this.getAlertColumns()
        columnName.forEach((item, index) => {
          item.children.forEach((child) => {
            child.cellStyle = (params) => {
              if (params.data.isAlert && !["AC_AL", "PN_AL", "RC_AL", "RE_AL", "TL_AL", "Month"].includes(params.colDef.field)) {
                return { "text-decoration": "none" }
              } else if(params.data.Month === "Total") {
                return { "text-decoration": "none" }
              }
            }
          })
        })

        return [
          {
            columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return alertMonthTable;
  }

  // /alertdrilldown?getYear=2022&gettingMonth=November&selectedRouteGroup=Network&selectedRegion=*&selectedCountry=*&selectedCity=*&getCabinValue=Null&type=Null&selectedRoute=*
  getAlertDrillDownData(
    year,
    month,
    routeGroup,
    region,
    country,
    city,
    route,
    cabinValue,
    type,
    priority = 'Null'
  ) {
    let selectedType = type;
    const selectedPriority = priority === 'Null' ? "Null" : `'${priority}'`
    // if (commonOD !== "*") {
    //   if (type == "Ancillary" || type == "Agency") {
    //     selectedType = "Null";
    //   }
    // }

    const url = `${API_URL}/alertdrilldown?getYear=${year}&gettingMonth=${month}&${AlertParams(
      routeGroup,
      region,
      country,
      city,
      route,
      cabinValue,
    )}&type=${selectedType}&priority=${selectedPriority}`;

    // const downloadUrl = `${API_URL}/FullYearDownloadPOS?getYear=${year}&${Params(
    //   regionId,
    //   countryId,
    //   cityId,
    //   getCabinValue
    // )}&commonOD=${String.addQuotesforMultiSelect(
    //   commonOD
    // )}&type=${selectedType}`;
    // localStorage.setItem("postype", type);
    // localStorage.setItem("posDownloadURL", downloadUrl);

    var alertDrillDownTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;


        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            actionName: key.ColumnName,
            isAlert: type === "Alert"
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            firstColumnName: "Total",
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
          });
        });

        return [
          {
            columnName: this.getAlertColumns("drilldown", firstColumnName, type),
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return alertDrillDownTable;
  }

  getAlertColumns(section, firstColumnName, type) {
    const isSortable = section === "drilldown" && type !== "Alert";
    const underline = section !== "drilldown";
    return [
      {
        headerName: "",
        children: [
          section === "drilldown" ? {
            headerName: firstColumnName,
            field: "firstColumnName",
            tooltipField: "firstColumnName",
            width: 250,
            alignLeft: true,
            underline: type === "Null",
          } : {
            headerName: string.columnName.MONTH,
            field: "Month",
            tooltipField: "Month",
            width: 150,
            underline: true,
            alignLeft: true,
          },
        ],
      },
      {
        headerName: string.columnName.REVENUE_$,
        children: [
          {
            headerName: "Actual",
            field: "CY_R",
            tooltipField: "CY_R_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          }, {
            headerName: "Var",
            field: "Var_R",
            tooltipField: "Var_R_AB",
            width: 150,
            sortable: isSortable,
            sort: isSortable && "asc",
            comparator: this.customSorting,
          },
          {
            headerName: "Var (%)",
            field: "VRev_R",
            tooltipField: "VRev_R_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },

        ],
      },
      {
        headerName: "Passenger-(O&D)",
        children: [
          {
            headerName: "Actual",
            field: "CY_P",
            tooltipField: "CY_P_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var",
            field: "Var_P",
            tooltipField: "Var_P_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var (%)",
            field: "VPax_P",
            tooltipField: "VPax_P_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          }
        ],
      },
      {
        headerName: "Booking",
        children: [
          {
            headerName: "Actual",
            field: "CY_B",
            tooltipField: "CY_B_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var",
            field: "Var_B",
            tooltipField: "Var_B_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var %",
            field: "VBook_B",
            tooltipField: "VBook_B_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.AVERAGE_FARE_$,
        children: [
          {
            headerName: "Actual",
            field: "CY_A",
            tooltipField: "CY_A_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var",
            field: "Var_A",
            tooltipField: "Var_A_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var (%)",
            field: "VAvg_A",
            tooltipField: "VAvg_A_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          }
        ],
      },
      {
        headerName: "Market Share",
        children: [
          {
            headerName: "MS (%)",
            field: "AL_MS",
            headerTooltip: "Airlines Market Size Growth",
            tooltipField: "AL_MS_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "MH MS (%)",
            field: "MH_MS",
            headerTooltip: "MH Market Size Growth",
            tooltipField: "MH_MS_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var",
            field: "Var_MS",
            tooltipField: "Var_MS_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          }
        ],
      },
      {
        headerName: "Price",
        children: [
          {
            headerName: "AL Fare",
            headerTooltip: "AL Fare",
            field: "AL_F",
            tooltipField: "AL_F_AB",
            width: 150,
            underline: !type || type === "Null",
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "MH Fare",
            headerTooltip: "MH Fare",
            field: "MH_F",
            tooltipField: "MH_F_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Var",
            field: "Var_PR",
            tooltipField: "Var_PR_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
        ],
      },
      {
        headerName: "Seats",
        children: [
          {
            headerName: "Avail (%)",
            headerTooltip: "Avail (%)",
            field: "AV_S",
            tooltipField: "AV_S_AB",
            width: 150,
            underline,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "Load Factor (%)",
            headerTooltip: "Load Factor (%)",
            field: "LF_S",
            tooltipField: "LF_S_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
          {
            headerName: "ASK (%)",
            headerTooltip: "ASK (%)",
            field: "ASK_S",
            tooltipField: "ASK_S_AB",
            width: 150,
            sortable: isSortable,
            comparator: this.customSorting,
          },
        ],
      },
      {
        headerName: "Alert Statistics",
        children: [
          {
            headerName: "A",
            headerTooltip: "Actioned",
            field: "AC_AL",
            tooltipField: "AC_AL_AB",
            width: 75,
            underline,
            sortable: isSortable,
            comparator: this.numberSorting,
          },
          {
            headerName: "P",
            headerTooltip: "Pending",
            field: "PN_AL",
            tooltipField: "PN_AL_AB",
            width: 75,
            underline,
            sortable: isSortable,
            comparator: this.numberSorting,
          },
          {
            headerName: "RC",
            headerTooltip: "Recurring",
            field: "RE_AL",
            tooltipField: "RE_AL_AB",
            width: 75,
            underline,
            sortable: isSortable,
            comparator: this.numberSorting,
          },
          {
            headerName: "RE",
            headerTooltip: "Rejected",
            field: "RJ_AL",
            tooltipField: "RJ_AL_AB",
            width: 75,
            underline,
            sortable: isSortable,
            comparator: this.numberSorting,
          },
          {
            headerName: "T",
            headerTooltip: "Total",
            field: "TL_AL",
            tooltipField: "TL_AL_AB",
            width: 75,
            underline,
            sortable: isSortable,
            comparator: this.numberSorting,
          },
        ],
      },
    ];

  }

  getActionDrillDown(
    year,
    month,
    routeGroup,
    region,
    country,
    city,
    route,
    cabinValue,
    action,
    type,
    alertType
  ) {
    const url = `${API_URL}/actionDrillDown?getYear=${year}&gettingMonth=${month}&${AlertParams(
      routeGroup,
      region,
      country,
      city,
      route,
      cabinValue,
    )}&getAction='${action}'&type='${type}'&alertType=${alertType}`;

    var alertDrillDownTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;


        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName,
            Month: key.ColumnName === null ? "---" : key.ColumnName,
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            isAlert: true,
          });
        });
        const priorityArray = ["High", "Moderate", "Low"]

        const tempArray = [...priorityArray]
        rowData.forEach((i) => {
          const index = tempArray.indexOf(i.Month);
          if (index > -1) {
            tempArray.splice(index, 1);
          }
        })
        if (rowData.length !== 0) {
          tempArray.forEach((priority) => {
            const last = { ...rowData[0] }
            Object.entries(last).forEach(([k]) => {
              last[k] = 0
            })
            last.Month = priority
            last.firstColumnName = priority
            last.isAlert = true
            rowData.push(last)
          })

          rowData.sort((a, b) => {
            const firstPriority = priorityArray.indexOf(a.Month);
            const secPriority = priorityArray.indexOf(b.Month)
            return firstPriority - secPriority
          })
        }
        return [
          {
            columnName: this.getAlertColumns("drilldown", firstColumnName, type),
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return alertDrillDownTable;

  }

  getActionDataDrillDown(
    year,
    month,
    routeGroup,
    region,
    country,
    city,
    route,
    cabinValue,
    alertType,
    action,
    priority
  ) {
    const url = `${API_URL}/actionDataDrillDown?getYear=${year}&gettingMonth=${month}&${AlertParams(
      routeGroup,
      region,
      country,
      city,
      route,
      cabinValue,
    )}&alertType=${alertType ?? "Null"}&getAction=${action ?? "Null"}&Priority=${priority && priority !== "Null" ? `'${priority}'` : "Null"}`;

    const columnName = city === "*" ? "POS" : route === "*" ? "OD" : "Cabin"

    var actionDataDrillDown = axios
      .get(url, this.getDefaultHeader()).then((response) => {

        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB:
              window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            isAlert: true
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            firstColumnName: "Total",
            CY_R: this.convertZeroValueToBlank(key.Revenue_CY),
            Var_R: this.convertZeroValueToBlank(key.Revenue_Var),
            VRev_R:
              this.convertZeroValueToBlank(key.Revenue_VRev),
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            Var_B: this.convertZeroValueToBlank(key.Bookings_Var),
            VBook_B: `${this.convertZeroValueToBlank(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P: this.convertZeroValueToBlank(key.Passenger_CY),
            Var_P: this.convertZeroValueToBlank(key.Passenger_Var),
            VPax_P: this.convertZeroValueToBlank(key.Passenger_VPax),
            CY_A: this.convertZeroValueToBlank(key.AvgFare_CY),
            Var_A: this.convertZeroValueToBlank(key.AvgFare_Var),
            VAvg_A:
              this.convertZeroValueToBlank(key.AvgFare_VAvg),
            AL_MS: this.convertZeroValueToBlank(key.AL_MS_Growth),
            MH_MS: this.convertZeroValueToBlank(key.MH_MS_Growth),
            Var_MS: this.convertZeroValueToBlank(key.MarketShare_Var),
            AL_F: this.convertZeroValueToBlank(key.AL_Fare),
            MH_F: this.convertZeroValueToBlank(key.MH_Fare),
            Var_PR: this.convertZeroValueToBlank(key.Price_Var),
            AV_S: this.convertZeroValueToBlank(key.Avail),
            LF_S: this.convertZeroValueToBlank(key.Load_Factor),
            ASK_S: this.convertZeroValueToBlank(key.ASK),
            TL_AL: key.Total_Alert,
            AC_AL: key.Actioned,
            RJ_AL: key.Rejected,
            RE_AL: key.Reccuring,
            PN_AL: key.Pending,
            CY_R_AB: window.numberWithCommas(key.Revenue_CY),
            Var_R_AB: window.numberWithCommas(key.Revenue_Var),
            VRev_R_AB:
              window.numberWithCommas(key.Revenue_VRev),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            Var_B_AB: window.numberWithCommas(key.Bookings_Var),
            VBook_B_AB: `${window.numberWithCommas(
              key.Bookings_VBook
            )}${this.showPercent(key.Bookings_VBook)}`,
            CY_P_AB: window.numberWithCommas(key.Passenger_CY),
            Var_P_AB: window.numberWithCommas(key.Passenger_Var),
            VPax_P_AB: window.numberWithCommas(key.Passenger_VPax),
            CY_A_AB: window.numberWithCommas(key.AvgFare_CY),
            Var_A_AB: window.numberWithCommas(key.AvgFare_Var),
            VAvg_A_AB: window.numberWithCommas(key.AvgFare_VAvg),
            AL_MS_AB: window.numberWithCommas(key.AL_MS_Growth),
            MH_MS_AB: window.numberWithCommas(key.MH_MS_Growth),
            Var_MS_AB: window.numberWithCommas(key.MarketShare_Var),
            AL_F_AB: window.numberWithCommas(key.AL_Fare),
            MH_F_AB: window.numberWithCommas(key.MH_Fare),
            Var_PR_AB: window.numberWithCommas(key.Price_Var),
            AV_S_AB: window.numberWithCommas(key.Avail),
            LF_S_AB: window.numberWithCommas(key.Load_Factor),
            ASK_S_AB: window.numberWithCommas(key.ASK),
            TL_AL_AB: key.Total_Alert,
            AC_AL_AB: key.Actioned,
            RJ_AL_AB: key.Rejected,
            PN_AL_AB: key.Pending,
            isAlert: key.is_alert,
          });
        });

        return [
          {
            columnName: this.getAlertColumns("drilldown", columnName, "Null"),
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: columnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });
    return actionDataDrillDown
  }

  getBookingTable(
    year,
    month,
    region,
    country,
    city,
    route,
    cabinValue) {
    const url = `${API_URL}/bookingtable?getYear=${year}&gettingMonth=${month}&${AlertParams(
      "Network",
      region,
      country,
      city,
      route,
      cabinValue
    )}`;
    var cabinTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.RBD,
            field: "RBD",
            tooltipField: "RBD_AB",
            alignLeft: true,
          },
          {
            headerName: "Bookings",
            field: "Booking",
            tooltipField: "Booking_AB",
          },
          {
            headerName: "VLY(%)",
            field: "VLY(%)",
            tooltipField: "VLY(%)_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
          {
            headerName: "Ticketed Average Fare(SR)",
            field: "Ticketed Average Fare(SR)",
            tooltipField: "Ticketed Average Fare(SR)_AB",
          },
          {
            headerName: "VLY(%)TKT",
            field: "VLY(%)TKT",
            tooltipField: "VLY(%)TKT_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
        ];

        var F = response.data.Data.filter((d) => d.Cabin === "F");
        var J = response.data.Data.filter((d) => d.Cabin === "J");
        var Y = response.data.Data.filter((d) => d.Cabin === "Y");

        var Total_F = response.data.Total.filter((d) => d.RBD === "Total of F");
        var Total_J = response.data.Total.filter((d) => d.RBD === "Total of J");
        var Total_Y = response.data.Total.filter((d) => d.RBD === "Total of Y");

        var mergedCabinData = [
          ...Total_F,
          ...F,
          ...Total_J,
          ...J,
          ...Total_Y,
          ...Y,
        ];
        var cabinData = [];

        mergedCabinData.forEach((key) => {
          cabinData.push({
            Cabin: key.Cabin,
            RBD: key.RBD,
            Booking: this.convertZeroValueToBlank(key.Bookings_CY),
            "VLY(%)": this.convertZeroValueToBlank(key.Bookings_VLY),
            "Ticketed Average Fare(SR)": this.convertZeroValueToBlank(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT": this.convertZeroValueToBlank(key.TicketedAverage_VLY),
            Booking_AB: window.numberWithCommas(key.Bookings_CY),
            "VLY(%)_AB": window.numberWithCommas(key.Bookings_VLY),
            "Ticketed Average Fare(SR)_AB": window.numberWithCommas(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT_AB": window.numberWithCommas(key.TicketedAverage_VLY),
          });
        });

        const totalData = []
        console.log(response.data);
        Array(response.data["Total_CY"]).forEach((key) => {
          totalData.push({
            Cabin: "Total",
            RBD: "Total",
            Booking: this.convertZeroValueToBlank(key.Bookings_CY),
            "VLY(%)": this.convertZeroValueToBlank(key.Bookings_VLY),
            "Ticketed Average Fare(SR)": this.convertZeroValueToBlank(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT": this.convertZeroValueToBlank(key.TicketedAverage_VLY),
            Booking_AB: window.numberWithCommas(key.Bookings_CY),
            "VLY(%)_AB": window.numberWithCommas(key.Bookings_VLY),
            "Ticketed Average Fare(SR)_AB": window.numberWithCommas(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT_AB": window.numberWithCommas(key.TicketedAverage_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            cabinData: cabinData,
            totalData: totalData
          },
        ]
      })
    return cabinTable;
  }

  //POS Page API
  getPOSMonthTables(
    currency,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue
  ) {
    const url = `${API_URL}/posDataMonthly?${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(commonOD)}`;

    var posmonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                underline: true,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
              },
            ],
          },
          {
            headerName: "Passenger-(O&D)",
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_P",
                tooltipField: "FRCT/Act_P_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_A",
                tooltipField: "FRCT/Act_A_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_R",
                tooltipField: "FRCT/Act_R_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_AL",
                tooltipField: "CY_AL_AB",
                underline: true,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_AL",
                tooltipField: "VLY_AL_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
        ];

        let previosYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear - 1
        );
        let currentYearTableDta = response.data.TableData.filter(
          (d) => d.Year === currentYear
        );
        let nextYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear + 1
        );

        var responseData = [
          ...response.data.Total_LY,
          ...previosYearTableData,
          ...currentYearTableDta,
          ...response.data.Total_NY,
          ...nextYearTableData,
        ];

        var rowData = [];

        responseData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        var totalData = [];
        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return posmonthtable;
  }

  getPOSDrillDownData(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    type,
    odsearchvalue
  ) {
    let selectedType = type;
    if (commonOD !== "*") {
      if (type == "Ancillary" || type == "Agency") {
        selectedType = "Null";
      }
    }

    const url = `${API_URL}/posDataDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&type=${type}&odSearch=${odsearchvalue}`;

    const downloadUrl = `${API_URL}/FullYearDownloadPOS?getYear=${getYear}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${String.addQuotesforMultiSelect(
      commonOD
    )}&type=${selectedType}`;
    localStorage.setItem("postype", type);
    localStorage.setItem("posDownloadURL", downloadUrl);

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;

        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  (type === "Null" || type === "Agency") &&
                    firstColumnName !== "Cabin"
                    ? true
                    : false,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: '', field: '', cellRenderer: (params) => this.alerts(params), width: 150 }]
          // },
          {
            headerName: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "Passenger-(O&D)",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_P",
                tooltipField: "FRCT/Act_P_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_A",
                tooltipField: "FRCT/Act_A_AB",
                width: 250,
                underline:
                  type === "Null" || firstColumnName === "Cabin" ? true : false,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_R",
                tooltipField: "FRCT/Act_R_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.AVAIL,
                field: "Avail",
                tooltipField: "Avail_AB",
                hide: firstColumnName === "Ancillary",
                underline:
                  type === "Null" || type === "OD" || type === "Cabin"
                    ? true
                    : false,
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_AL",
                tooltipField: "CY_AL_AB",
                hide: firstColumnName === "Ancillary",
                underline: type === "Null" || type === "OD" ? true : false,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_AL",
                tooltipField: "VLY_AL_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            "": "",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            Avail_AB: window.numberWithCommas(key.Avail),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            isAlert: key.is_alert,
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            Avail_AB: window.numberWithCommas(key.Avail),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            isAlert: key.is_alert,
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return posregiontable;
  }

  getPOSCabinDetails(
    getYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue
  ) {
    const url = `${API_URL}/poscabinWiseDetails?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(commonOD)}`;

    var cabinTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.RBD,
            field: "RBD",
            tooltipField: "RBD_AB",
            alignLeft: true,
          },
          {
            headerName: "Bookings",
            field: "Booking",
            tooltipField: "Booking_AB",
          },
          {
            headerName: "VLY(%)",
            field: "VLY(%)",
            tooltipField: "VLY(%)_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
          {
            headerName: "Ticketed Average Fare(SR)",
            field: "Ticketed Average Fare(SR)",
            tooltipField: "Ticketed Average Fare(SR)_AB",
          },
          {
            headerName: "VLY(%)TKT",
            field: "VLY(%)TKT",
            tooltipField: "VLY(%)TKT_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
        ];

        var F = response.data.Data.filter((d) => d.Cabin === "F");
        var J = response.data.Data.filter((d) => d.Cabin === "J");
        var Y = response.data.Data.filter((d) => d.Cabin === "Y");

        var Total_F = response.data.Total.filter((d) => d.RBD === "Total of F");
        var Total_J = response.data.Total.filter((d) => d.RBD === "Total of J");
        var Total_Y = response.data.Total.filter((d) => d.RBD === "Total of Y");

        var mergedCabinData = [
          ...Total_F,
          ...F,
          ...Total_J,
          ...J,
          ...Total_Y,
          ...Y,
        ];
        var cabinData = [];

        mergedCabinData.forEach((key) => {
          cabinData.push({
            Cabin: key.Cabin,
            RBD: key.RBD,
            Booking: this.convertZeroValueToBlank(key.Bookings_CY),
            "VLY(%)": this.convertZeroValueToBlank(key.Bookings_VLY),
            "Ticketed Average Fare(SR)": this.convertZeroValueToBlank(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT": this.convertZeroValueToBlank(key.TicketedAverage_VLY),
            Booking_AB: window.numberWithCommas(key.Bookings_CY),
            "VLY(%)_AB": window.numberWithCommas(key.Bookings_VLY),
            "Ticketed Average Fare(SR)_AB": window.numberWithCommas(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT_AB": window.numberWithCommas(key.TicketedAverage_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            cabinData: cabinData,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return cabinTable;
  }

  getAlertSummary(alertId) {
    const url = `${API_URL}/alertsummary?alertId='${alertId}'`;

    const alertSummary = axios.get(url, this.getDefaultHeader()).then((response) => {
      return response.data.response.response;
    })
    return alertSummary
  }

  // /alertdetails?selectedRouteGroup=Network&selectedRegion='South Asia'&selectedCountry='NP'&selectedRoute='BNEKTM'&getCabinValue=Null&selectedCity='KTM'&getCabinValue='Y'&getYear=2022&gettingMonth=November
  getAlertDetails(year,
    month,
    region,
    country,
    city,
    route,
    cabinValue) {
    const url = `${API_URL}/alertdetails?getYear=${year}&gettingMonth=${month}&${AlertDetailsParams(
      "Network",
      region,
      country,
      city,
      route,
      cabinValue
    )}`;

    const alertDetails = axios.get(url, this.getDefaultHeader()).then((response) => {
      var columnName = [
        [
          {
            headerName: "Name",
            field: "Name",
            width: 150,
            alignLeft: true
          },
          {
            headerName: "Actual",
            field: "CY",
            tooltipField: "CY_AB",
            width: 150,
          },
          {
            headerName: "Target",
            field: "TGT",
            tooltipField: "TGT_AB",
            width: 150,
          },
          {
            headerName: "Var",
            field: "Var",
            tooltipField: "Var_AB",
            width: 150,
          },
          {
            headerName: "Var%",
            field: "VarP",
            tooltipField: "varP_AB",
            width: 150,
          },
          {
            headerName: "Status",
            field: "Status",
            width: 150,
            alignLeft: true,
            cellRenderer: (params) => this.statusArrowIndicator(params),
          },
        ],
        [
          {
            headerName: "Name",
            field: "Name",
            width: 150,
            alignLeft: true
          },
          {
            headerName: "AL MS",
            field: "AL_MS",
            tooltipField: "AL_MS_AB",
            width: 150,
          },
          {
            headerName: "MH AL MS",
            field: "MH_AL_MS",
            tooltipField: "MH_AL_MS_AB",
            width: 150,
          },
          {
            headerName: "Var",
            field: "Var_MS",
            tooltipField: "Var_MS_AB",
            width: 150,
          },
          {
            headerName: "Status",
            field: "Status",
            width: 150,
            alignLeft: true,
            cellRenderer: (params) => this.statusArrowIndicator(params),
          },
        ],
        [
          {
            headerName: "Name",
            field: "Name",
            width: 150,
            alignLeft: true
          },
          {
            headerName: "MH Fare",
            field: "MH_F",
            tooltipField: "MH_F_AB",
            width: 150,
          },
          {
            headerName: "AL Fare",
            field: "AL_F",
            tooltipField: "AL_F_AB",
            width: 150,
          },
          {
            headerName: "Var",
            field: "Var_P",
            tooltipField: "Var_P_AB",
            width: 150,
          },
          {
            headerName: "Status",
            field: "Status",
            width: 150,
            alignLeft: true,
            cellRenderer: (params) => this.statusArrowIndicator(params),
          },
        ],
        [
          {
            headerName: "Name",
            field: "Name",
            width: 150,
            alignLeft: true
          },
          {

            headerName: "ASK",
            field: "ASK_S",
            tooltipField: "ASK_S_AB",
            width: 150,
          },
          {
            headerName: "Avail",
            field: "AV_S",
            tooltipField: "AV_S_AB",
            width: 150,
          },
          {
            headerName: "Load Factor",
            field: "LF_S",
            tooltipField: "LF_S_AB",
            width: 150,
          },
          {
            headerName: "Status",
            field: "Status",
            width: 150,
            alignLeft: true,
            cellRenderer: (params) => this.statusArrowIndicator(params),
          },
        ],
      ];

      const rowData = [[], [], [], []]
      response.data.TableData.forEach((key) => {
        Object.entries(key).forEach(([k, v]) => {
          if (k === "Id" || k === "Flag") return
          rowData[0].push({
            Name: v.Name,
            Status: v.Status,
            CY: this.convertZeroValueToBlank(v.CY),
            TGT: this.convertZeroValueToBlank(v.TGT),
            Var: this.convertZeroValueToBlank(v.Var),
            VarP: this.convertZeroValueToBlank(v["Var%"]),
          }
          )
        })
      })

      // Market Share
      response.data.MS_Data.forEach((key) => {
        rowData[1].push({
          Name: key.Name,
          Status: key.Status,
          AL_MS: this.convertZeroValueToBlank(key.All_MS_CY),
          MH_AL_MS: this.convertZeroValueToBlank(key.All_MS_LY),
          Var_MS: this.convertZeroValueToBlank(key.Var),
          AL_MS_AB: window.numberWithCommas(key.All_MS_CY),
          MH_AL_AB: window.numberWithCommas(key.All_MS_LY),
          Var_MS_AB: window.numberWithCommas(key.Var),
        })
      })

      // Infare
      response.data.Infare.forEach((key) => {
        rowData[2].push({
          Name: key.Name,
          Status: key.Status,
          MH_F: this.convertZeroValueToBlank(key.MH_Fare),
          AL_F: this.convertZeroValueToBlank(key.AL_Fare),
          Var_P: this.convertZeroValueToBlank(key.Var),
          MH_F_AB: window.numberWithCommas(key.MH_Fare),
          AL_F_AB: window.numberWithCommas(key.Comp_Fare),
          Var_P_AB: window.numberWithCommas(key.Var),
        })
      })

      // Availability
      response.data.Availability.forEach((key) => {
        rowData[3].push({
          Name: key.Name,
          Status: key.Status,
          ASK_S: this.convertZeroValueToBlank(key.ASK),
          AV_S: this.convertZeroValueToBlank(key.Avail),
          LF_S: this.convertZeroValueToBlank(key.Load_Factor),
          ASK_S_AB: window.numberWithCommas(key.ASK),
          AV_S_AB: window.numberWithCommas(key.Avail),
          LF_S_AB: window.numberWithCommas(key.Load_Factor),
        })
      })
      return {
        id: response.data.TableData[0]?.Id,
        flag: response.data.TableData[0]?.Flag,
        columnName,
        rowData
      }
    })
    return alertDetails
  }

  postAlertDetails(
    region,
    country,
    city,
    route,
    cabinValue,
    action,
    Id,
    Message) {
    let user = JSON.parse(cookieStorage.getCookie("userDetails"));

    const url = `${API_URL}/alertdetails?${AlertDetailsParams(
      "Network",
      region,
      country,
      city,
      route,
      cabinValue
    )}`;
    return axios.post(url, {
      action,
      Id,
      Message,
      Username: user.username,
      Date: new Date().toISOString().slice(0, 10)
    }, this.getDefaultHeader())
  }

  getInfareTrendData(
    year,
    month,
    region,
    country,
    city,
    route,
    cabinValue,
    type,
  ) {
    const url = `${API_URL}/trendTable?getYear=${year}&gettingMonth=${month}&selectedRouteGroup=Network&selectedRegion=${region}&selectedCountry=${country}&selectedCity=${city}&selectedRoute=${route}&getCabinValue=${cabinValue}&getTrend='${type}'`;

    return axios.get(url, this.getDefaultHeader()).then((response) => {
      var priorityArray = ["High", "Moderate", "Low", 0]

      response.data.TableData.sort((a, b) => {
        var firstPriority = priorityArray.indexOf(a.Alert_Type);
        var secPriority = priorityArray.indexOf(b.Alert_Type)
        return secPriority - firstPriority
      })
      return response.data
    })
      .catch((error) => {
        console.log(error);
      });
  }

  // /trendTable?selectedRouteGroup=Network&selectedRegion='Europe'&selectedCountry=*&selectedRoute=*&selectedCity=*&getCabinValue='y'&getYear=2022&gettingMonth=November&getTrend='Revenue'
  getAlertTrendData(
    year,
    month,
    region,
    country,
    city,
    route,
    cabinValue,
    type,
  ) {
    const url = `${API_URL}/trendTable?getYear=${year}&gettingMonth=${month}&${AlertParams(
      "Network",
      region,
      country,
      city,
      route,
      cabinValue
    )}&getTrend='${type}'`;


    const trendData = axios.get(url, this.getDefaultHeader()).then((response) => {
      let columnName = []
      if (type === "Infare") {
        columnName = [
          {
            headerName: "Priority",
            field: "Alert_Type",
          },
          {
            headerName: "MH Fare",
            field: "MH_F",
            tooltipField: "MH_F_AB"
          },
          {
            headerName: "Comp Fare",
            field: "CMP_F",
            tooltipField: "CMP_F_AB",
          },
          {
            headerName: "Rank",
            field: "RANK",
            tooltipField: "RANK_AB",
          }
        ]
      } else if (type === "Avail") {
        columnName = [
          {
            headerName: "Priority",
            field: "Alert_Type",
          },
          {
            headerName: "Avail (%)",
            field: "AV_AV",
            tooltipField: "AV_AV_AB"
          },
          {
            headerName: "Load Factor (%)",
            field: "LF_AV",
            tooltipField: "LF_AV_AB",
          },
          {
            headerName: "ASK (%)",
            field: "ASK_AV",
            tooltipField: "ASK_AV_AB",
          }
        ]
      } else if (type === "MarketShare") {
        columnName = [
          {
            headerName: "AL",
            field: "AL",
            tooltipField: "AL_AB",
            sortable: true
          },
          {
            headerName: "AL MS",
            field: "AL_MS",
            tooltipField: "AL_MS_AB",
            sortable: true
          },
          {
            headerName: "MH AL MS",
            field: "MH_AL_MS",
            tooltipField: "MH_AL_MS_AB",
            sortable: true
          },
          {
            headerName: "AL MS Growth",
            field: "AL_MS_G",
            tooltipField: "AL_MS_G_AB",
            sortable: true
          },
          {
            headerName: "MH MS Growth",
            field: "MH_MS_G",
            tooltipField: "MH_MS_G_AB",
            sortable: true
          }, {
            headerName: "Var Growth",
            field: "Var_G",
            tooltipField: "Var_G_AB",
            sortable: true
          },
        ]
      } else {
        columnName = [
          {
            headerName: "Priority",
            field: "Alert_Type",
          },
          {
            headerName: "Actual",
            field: "CY",
            tooltipField: "CY_AB",
          },
          {
            headerName: "LY",
            field: "TGT",
            tooltipField: "TGT_AB",
          },
          {
            headerName: "Var",
            field: "Var",
            tooltipField: "Var_AB",
          },
          {
            headerName: "Var%",
            field: "VarP",
            tooltipField: "VarP_AB",
          },
        ]
      }

      const rowData = []
      const getRow = (key, isTotal) => {
        return {
          Alert_Type: key.Alert_Type ?? (isTotal ? "Total" : ""),
          Status: key.Status,
          CY: this.convertZeroValueToBlank(key.CY),
          TGT: this.convertZeroValueToBlank(key.TGT),
          Var: this.convertZeroValueToBlank(key.Var),
          VarP: this.convertZeroValueToBlank(key["Var%"]),
          CY_AB: window.numberWithCommas(key.CY),
          TGT_AB: window.numberWithCommas(key.TGT),
          Var_AB: window.numberWithCommas(key.Var),
          VarP_AB: window.numberWithCommas(key["Var%"]),

          // For Market Share
          //AL: this.convertZeroValueToBlank(key.AL),
          AL: key.AL ?? "Total",
          AL_MS: this.convertZeroValueToBlank(key.AL_MS),
          MH_AL_MS: this.convertZeroValueToBlank(key.MH_AL_MS),
          AL_MS_G: this.convertZeroValueToBlank(key.AL_MS_Growth),
          MH_MS_G: this.convertZeroValueToBlank(key.MH_MS_Growth),
          Var_G: this.convertZeroValueToBlank(key.Var_Growth),
          //AL_AB: window.numberWithCommas(key.AL),
          AL_MS_AB: window.numberWithCommas(key.AL_MS),
          MH_AL_MS_AB: window.numberWithCommas(key.MH_AL_MS),
          AL_MS_G_AB: window.numberWithCommas(key.AL_MS_Growth),
          MH_MS_G_AB: window.numberWithCommas(key.MH_MS_Growth),
          Var_G_AB: window.numberWithCommas(key.Var_Growth),

          // For Infare
          MH_F: this.convertZeroValueToBlank(key.MH_Fare),
          CMP_F: this.convertZeroValueToBlank(key.Comp_Fare),
          RANK: this.convertZeroValueToBlank(key.Rank),
          MH_F_AB: window.numberWithCommas(key.MH_Fare),
          CMP_F_AB: window.numberWithCommas(key.Comp_Fare),
          RANK_AB: window.numberWithCommas(key.Rank),

          // For Availability
          AV_AV: this.convertZeroValueToBlank(key.Avail),
          LF_AV: this.convertZeroValueToBlank(key.Load_Factor),
          ASK_AV: this.convertZeroValueToBlank(key.ASK),
          AV_AV_AB: window.numberWithCommas(key.Avail),
          LF_AV_AB: window.numberWithCommas(key.Load_Factor),
          ASK_AV_AB: window.numberWithCommas(key.ASK),
        }
      }
      response.data.TableData.forEach((key) => {
        rowData.push(getRow(key))
      })
      var priorityArray = ["High", "Moderate", "Low", 0]

      rowData.sort((a, b) => {
        var firstPriority = priorityArray.indexOf(a.Alert_Type);
        var secPriority = priorityArray.indexOf(b.Alert_Type)
        return firstPriority - secPriority
      })

      rowData.push(getRow(response.data.Total, true))
      return [{
        columnName,
        rowData,
        name: response.data.TableData[0].Name
      }]
    })
    return trendData
  }

  getAvailabilityDetails(
    getYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    type,
    typeParameter
  ) {
    let that = this;
    const url = `${API_URL}/seatavailability?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(
      commonOD
    )}&type=${type}&typeParameter=${typeParameter}`;
    var availabilityData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.RBD,
            field: "RBD",
            tooltipField: "RBD_AB",
            alignLeft: true,
          },
          {
            headerName: "Availability",
            field: "Availability",
            tooltipField: "Availability_AB",
            alignLeft: true,
          },
        ];

        var F = response.data.Data.filter((d) => d.Cabin === "F");
        var J = response.data.Data.filter((d) => d.Cabin === "J");
        var Y = response.data.Data.filter((d) => d.Cabin === "Y");

        var Total_F = response.data.Total.filter((d) => d.RBD === "Total of F");
        var Total_J = response.data.Total.filter((d) => d.RBD === "Total of J");
        var Total_Y = response.data.Total.filter((d) => d.RBD === "Total of Y");

        var mergedCabinData = [
          ...Total_F,
          ...F,
          ...Total_J,
          ...J,
          ...Total_Y,
          ...Y,
        ];
        const rowData = [];
        mergedCabinData.forEach(function (key) {
          rowData.push({
            Cabin: key.Cabin,
            RBD: key.RBD,
            Availability: that.convertZeroValueToBlank(key.Availability),
            Availability_AB: window.numberWithCommas(key.Availability),
          });
        });

        return [{ columnName: columnName, rowData: rowData }];
      })
      .catch((error) => {
        that.errorHandling(error);
      });
    return availabilityData;
  }

  getPOSLineCharts(displayName, region, country, city, od, getCabin) {
    let link = "";

    if (displayName === string.columnName.BOOKINGS) {
      link = "posbooking";
    }
    if (displayName === "Passenger-(O&D)") {
      link = "pospassenger";
    }
    if (displayName === string.columnName.AVERAGE_FARE_$) {
      link = "posavgfare";
    }
    if (displayName === string.columnName.REVENUE_$) {
      link = "posrevenue";
    }
    const url = `${API_URL}/${link}?regionId=${region}&countryId=${country}&cityId=${city}&commonOD=${od}&getCabinValue=${getCabin}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getPOSLineChartsForecast(
    displayName,
    region,
    country,
    city,
    od,
    getCabin,
    gettingYear,
    gettingMonth
  ) {
    let link = "";

    if (displayName === string.columnName.BOOKINGS) {
      link = "posbooking";
    }
    if (displayName === "Passenger Forecast") {
      link = "posPassengerForeGraph";
    }
    if (displayName === "Average fare Forecast") {
      link = "posAvgFareForeGraph";
    }
    if (displayName === "Revenue Forecast") {
      link = "posRevenueForeGraph";
    }
    const url = `${API_URL}/${link}?getYear=${gettingYear}&gettingMonth=${gettingMonth}&regionId=${region}&countryId=${country}&cityId=${city}&commonOD=${od}&getCabinValue=${getCabin}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  exportCSVPOSMonthlyURL(regionId, countryId, cityId, commonOD, getCabinValue) {
    const url = `posDataMonthly?${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(commonOD)}`;
    return url;
  }

  exportCSVPOSDrillDownURL(
    getYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    type
  ) {
    const url = `posDataDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(commonOD)}&type=${type}`;
    return url;
  }

  //POS Promotion Trackin Page API
  getPOSPromotionMonthTables(
    currency,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue
  ) {
    const url = `${API_URL}/promoTrackMonthly?${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}`;
    var posmonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          // {
          //     headerName: string.columnName.BOOKINGS,
          //     headerGroupComponent: 'customHeaderGroupComponent',
          //     children: [
          //         { headerName: string.columnName.CY, field: 'CY_B', tooltipField: 'CY_B_AB', underline: true },
          //         { headerName: string.columnName.LY, field: 'LY_B', tooltipField: 'LY_B_AB' },
          //         { headerName: string.columnName.VLY, field: 'VLY_B', tooltipField: 'VLY_B_AB', cellRenderer: (params) => this.arrowIndicator(params) },
          //     ]
          // },
          {
            headerName: "Passenger-(O&D)",
            //  headerGroupComponent: 'customHeaderGroupComponent',
            children: [
              // {
              //     headerName: string.columnName.FORECAST_ACT, field: 'FRCT/Act_P', tooltipField: 'FRCT/Act_P_AB', width: 250,
              //     cellClassRules: {
              //         'align-right-underline': params => params.data.isUnderline
              //     }
              // },
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_P",
                tooltipField: "LY_P_AB",
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            //  headerGroupComponent: 'customHeaderGroupComponent',
            children: [
              // {
              //     headerName: string.columnName.FORECAST_ACT, field: 'FRCT/Act_A', tooltipField: 'FRCT/Act_A_AB', width: 250,
              //     cellClassRules: {
              //         'align-right-underline': params => params.data.isUnderline
              //     }
              // },
              {
                headerName: string.columnName.CY,
                field: "CY_A",
                tooltipField: "CY_A_AB",
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_A",
                tooltipField: "LY_A_AB",
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            // headerGroupComponent: 'customHeaderGroupComponent',
            children: [
              // {
              //     headerName: string.columnName.FORECAST_ACT, field: 'FRCT/Act_R', tooltipField: 'FRCT/Act_R_AB', width: 250,
              //     cellClassRules: {
              //         'align-right-underline': params => params.data.isUnderline
              //     }
              // },
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_R",
                tooltipField: "LY_R_AB",
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          // {
          //     headerName: string.columnName.AL_MARKET_SHARE,
          //     headerGroupComponent: 'customHeaderGroupComponent',
          //     children: [
          //         { headerName: string.columnName.CY, field: 'CY_AL', tooltipField: 'CY_AL_AB', underline: true },
          //         { headerName: string.columnName.VLY, field: 'VLY_AL', tooltipField: 'VLY_AL_AB', cellRenderer: (params) => this.arrowIndicator(params) }]
          // }
        ];

        let previosYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear - 1
        );
        let currentYearTableDta = response.data.TableData.filter(
          (d) => d.Year === currentYear
        );
        let nextYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear + 1
        );

        var responseData = [
          ...response.data.Total_LY,
          ...previosYearTableData,
          ...currentYearTableDta,
          ...response.data.Total_NY,
          ...nextYearTableData,
        ];

        var rowData = [];

        responseData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: this.convertZeroValueToBlank(key.Bookings_LY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            CY_P: this.convertZeroValueToBlank(key.Passenger_FRCT),
            LY_P: this.convertZeroValueToBlank(key.Passenger_LY),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            CY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            LY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_LY)
                : this.convertZeroValueToBlank(key.AvgFare_LY),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            LY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_LY)
                : this.convertZeroValueToBlank(key.Revenue_LY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            CY_P_AB: window.numberWithCommas(key.Passenger_FRCT),
            LY_P_AB: window.numberWithCommas(key.Passenger_LY),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            CY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            LY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_LY)
                : window.numberWithCommas(key.AvgFare_LY),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            LY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_LY)
                : window.numberWithCommas(key.Revenue_LY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        var totalData = [];
        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: this.convertZeroValueToBlank(key.Bookings_LY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            CY_P: this.convertZeroValueToBlank(key.Passenger_FRCT),
            LY_P: this.convertZeroValueToBlank(key.Passenger_LY),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            CY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            LY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_LY)
                : this.convertZeroValueToBlank(key.AvgFare_LY),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            LY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_LY)
                : this.convertZeroValueToBlank(key.Revenue_LY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            LY_B_AB: window.numberWithCommas(key.Bookings_LY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            CY_P_AB: window.numberWithCommas(key.Passenger_FRCT),
            LY_P_AB: window.numberWithCommas(key.Passenger_LY),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            CY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            LY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_LY)
                : window.numberWithCommas(key.AvgFare_LY),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            LY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_LY)
                : window.numberWithCommas(key.Revenue_LY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return posmonthtable;
  }

  getPOSPromotionDrillDownData(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/promoTrackDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&type=${type}`;

    localStorage.setItem("posPromotionDownloadURL", url);

    var posregiontable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.ColumnName;

        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  type === "Null" && firstColumnName !== "Cabin" ? true : false,
              },
            ],
          },
          {
            headerName: "Passenger-(O&D)",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_P",
                tooltipField: "CY_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_P",
                tooltipField: "LY_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_A",
                tooltipField: "CY_A_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_A",
                tooltipField: "LY_A_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_R",
                tooltipField: "LY_R_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            "": "",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: this.convertZeroValueToBlank(key.Bookings_LY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            CY_P: this.convertZeroValueToBlank(key.Passenger_FRCT),
            LY_P: this.convertZeroValueToBlank(key.Passenger_LY),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            CY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            LY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_LY)
                : this.convertZeroValueToBlank(key.AvgFare_LY),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            LY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_LY)
                : this.convertZeroValueToBlank(key.Revenue_LY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            LY_B_AB: window.numberWithCommas(key.Bookings_LY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            CY_P_AB: window.numberWithCommas(key.Passenger_FRCT),
            LY_P_AB: window.numberWithCommas(key.Passenger_LY),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            CY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            LY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_LY)
                : window.numberWithCommas(key.AvgFare_LY),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            LY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_LY)
                : window.numberWithCommas(key.Revenue_LY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            Avail_AB: window.numberWithCommas(key.Avail),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            isAlert: key.is_alert,
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            LY_B: this.convertZeroValueToBlank(key.Bookings_LY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            CY_P: this.convertZeroValueToBlank(key.Passenger_FRCT),
            LY_P: this.convertZeroValueToBlank(key.Passenger_LY),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            CY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            LY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_LY)
                : this.convertZeroValueToBlank(key.AvgFare_LY),
            VLY_A:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            LY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_LY)
                : this.convertZeroValueToBlank(key.Revenue_LY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            LY_B_AB: window.numberWithCommas(key.Bookings_LY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            CY_P_AB: window.numberWithCommas(key.Passenger_FRCT),
            LY_P_AB: window.numberWithCommas(key.Passenger_LY),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            CY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            LY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_LY)
                : window.numberWithCommas(key.AvgFare_LY),
            VLY_A_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            LY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_LY)
                : window.numberWithCommas(key.Revenue_LY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            Avail_AB: window.numberWithCommas(key.Avail),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            isAlert: key.is_alert,
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            currentAccess: response.data.CurrentAccess,
            totalData: totalData,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return posregiontable;
  }

  getPromoTitleGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/PromoTitleGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getTicketbyChannelGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/TicketbyChannelGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getChannelbyRevenueGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/ChannelbyRevenueGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getTop10NonDirectionalODGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/Top10Non_directionalODGraph?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getRevenueByIssueDateGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/RevenueByIssueDate?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getNoOfTicketsByIssueGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/NoOfTicketsByIssueDate?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getRevenueByTravelPeriodGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/RevenueByTravelPeriod?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  getNoOfTicketsByTravelPeriodGraph(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    serviceGroupId,
    promoTypeId,
    promoTitleId,
    agencyGroupId,
    agentsId,
    commonODId,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/NoOfTicketByTravelPeriod?getYear=${getYear}&gettingMonth=${gettingMonth}&${PromotionParams(
      regionId,
      countryId,
      serviceGroupId,
      promoTypeId,
      promoTitleId,
      agencyGroupId,
      agentsId,
      commonODId,
      getCabinValue
    )}&dataType=${type}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        this.errorHandling(error);
      });
  }

  // Cabin List
  getClassNameDetails() {
    const url = `${API_URL}/getClassDetails`;
    var classNameDetails = axios
      .get(url)
      .then(function (response) {
        var posClasstableDatas = [];
        response.data.forEach(function (key) {
          posClasstableDatas.push({
            ClassText: key.Cabin,
            ClassValue: key.Cabin,
          });
        });

        return [{ classDatas: posClasstableDatas }]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });
    return classNameDetails;
  }

  // Route Page API Data Processing
  getRouteMonthTables(
    currency,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/routeDataMonthly?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}`;
    var routemonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            headerTooltip: string.columnName.BOOKINGS,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                underline: true,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER_OD,
            headerTooltip: string.columnName.PASSENGER_OD,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_P",
                tooltipField: "FRCT/Act_P_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            headerTooltip: string.columnName.AVERAGE_FARE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_A",
                tooltipField: "FRCT/Act_A_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Avg",
                tooltipField: "VLY_Avg_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerTooltip: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_R",
                tooltipField: "FRCT/Act_R_AB",
                width: 250,
                cellClassRules: {
                  "align-right-underline": (params) => params.data.isUnderline,
                },
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Rev",
                tooltipField: "VLY_Rev_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          // {
          //     headerName: string.columnName.AL_MARKET_SHARE,
          //     children: [
          //         { headerName: string.columnName.CY, field: 'CY_AL', tooltipField: 'CY_AL' },
          //         { headerName: string.columnName.VLY, field: 'VLY_AL', tooltipField: 'VLY_AL', cellRenderer: (params) => this.arrowIndicator(params), width: 250 }]
          // },
          {
            headerName: string.columnName.RASK_DOLLAR,
            headerTooltip: string.columnName.RASK_DOLLAR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
              },
            ],
          },
          {
            headerName: string.columnName.ASK,
            headerTooltip: string.columnName.ASK,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_A",
                tooltipField: "CY_A_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
              },
            ],
          },
          {
            headerName: string.columnName.YIELD,
            headerTooltip: string.columnName.YIELD,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Y",
                tooltipField: "CY_Y_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Y",
                tooltipField: "VLY_Y_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
              },
            ],
          },
          {
            headerName: string.columnName.LOAD_FACTOR,
            headerTooltip: string.columnName.LOAD_FACTOR,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_L",
                tooltipField: "CY_L_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_L",
                tooltipField: "VLY_L_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
              },
            ],
          },
        ];

        let previosYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear - 1
        );
        let currentYearTableDta = response.data.TableData.filter(
          (d) => d.Year === currentYear
        );
        let nextYearTableData = response.data.TableData.filter(
          (d) => d.Year === currentYear + 1
        );

        var responseData = [
          ...response.data.Total_LY,
          ...previosYearTableData,
          ...currentYearTableDta,
          ...response.data.Total_NY,
          ...nextYearTableData,
        ];

        var routemonthtableDatas = [];

        responseData.forEach((key) => {
          routemonthtableDatas.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            "": "",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_Avg:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_Rev:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_CY)
                : this.convertZeroValueToBlank(key.Rask_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_VLY)
                : this.convertZeroValueToBlank(key.Rask_VLY),
            CY_A: this.convertZeroValueToBlank(key.ASK_CY),
            VLY_A: this.convertZeroValueToBlank(key.ASK_VLY),
            CY_Y: this.convertZeroValueToBlank(key.yield_CY),
            VLY_Y: this.convertZeroValueToBlank(key.yield_VLY),
            CY_L: this.convertZeroValueToBlank(key.LoadFactor_CY),
            VLY_L: this.convertZeroValueToBlank(key.LoadFactor_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_Avg_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_Rev_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_CY)
                : window.numberWithCommas(key.Rask_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_VLY)
                : window.numberWithCommas(key.Rask_VLY),
            CY_A_AB: window.numberWithCommas(key.ASK_CY),
            VLY_A_AB: window.numberWithCommas(key.ASK_VLY),
            CY_Y_AB: window.numberWithCommas(key.yield_CY),
            VLY_Y_AB: window.numberWithCommas(key.yield_VLY),
            CY_L_AB: window.numberWithCommas(key.LoadFactor_CY),
            VLY_L_AB: window.numberWithCommas(key.LoadFactor_VLY),
            Year: key.Year,
            MonthName: key.monthfullname,
            isUnderline:
              parseInt(key.Year) == currentYear
                ? key.MonthNumber >= currentMonth
                : parseInt(key.Year) > currentYear
                  ? key.MonthNumber < currentMonth
                  : false,
          });
        });

        var totalData = [];

        response.data.Total_CY.forEach((key) => {
          totalData.push({
            Month: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_Avg:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_Rev:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_CY)
                : this.convertZeroValueToBlank(key.Rask_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_VLY)
                : this.convertZeroValueToBlank(key.Rask_VLY),
            CY_A: this.convertZeroValueToBlank(key.ASK_CY),
            VLY_A: this.convertZeroValueToBlank(key.ASK_VLY),
            CY_Y: this.convertZeroValueToBlank(key.yield_CY),
            VLY_Y: this.convertZeroValueToBlank(key.yield_VLY),
            CY_L: this.convertZeroValueToBlank(key.LoadFactor_CY),
            VLY_L: this.convertZeroValueToBlank(key.LoadFactor_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_Avg_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_Rev_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_CY)
                : window.numberWithCommas(key.Rask_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_VLY)
                : window.numberWithCommas(key.Rask_VLY),
            CY_A_AB: window.numberWithCommas(key.ASK_CY),
            VLY_A_AB: window.numberWithCommas(key.ASK_VLY),
            CY_Y_AB: window.numberWithCommas(key.yield_CY),
            VLY_Y_AB: window.numberWithCommas(key.yield_VLY),
            CY_L_AB: window.numberWithCommas(key.LoadFactor_CY),
            VLY_L_AB: window.numberWithCommas(key.LoadFactor_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            routemonthtableDatas: routemonthtableDatas,
            currentAccess: response.data.CurretAccess,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log("error", error);
      });

    return routemonthtable;
  }

  getRouteDrillDownData(
    getYear,
    currency,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/routeDataDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&type=${type}`;

    const downloadurl = `${API_URL}/FullYearDownloadRoute?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&type=${type}`;

    localStorage.setItem("routeDownloadURL", downloadurl);

    var routeRegionTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.TableData.filter(
          (d) => d.AvgFare_TGT === 0 || d.AvgFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.TableData.length;

        let revenuzeroTGT = response.data.TableData.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.TableData.length;

        let passengerzeroTGT = response.data.TableData.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.TableData.length;

        const firstColumnName = response.data.ColumnName;
        const isAncillary = firstColumnName === "Ancillary" ? true : false;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField:
                  firstColumnName === "Ancillary"
                    ? "Ancillary_Full_Name"
                    : "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  type === "Null" && firstColumnName !== "Cabin" ? true : false,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            headerTooltip: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                sortable: true,
                hide: firstColumnName === "Ancillary",
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER_OD,
            headerTooltip: string.columnName.PASSENGER_OD,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_P",
                tooltipField: "FRCT/Act_P_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            headerTooltip: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_A",
                tooltipField: "FRCT/Act_A_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Avg",
                tooltipField: "VLY_Avg_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerTooltip: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "FRCT/Act_R",
                tooltipField: "FRCT/Act_R_AB",
                width: 250,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Rev",
                tooltipField: "VLY_Rev_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [
          //         { headerName: string.columnName.AVAIL, field: 'Avail', tooltipField: 'Avail' }]
          // },
          // {
          //     headerName: string.columnName.AL_MARKET_SHARE,
          //     children: [
          //         { headerName: string.columnName.CY, field: 'CY_AL', tooltipField: 'CY_AL' },
          //         { headerName: string.columnName.VLY, field: 'VLY_AL', tooltipField: 'VLY_AL', cellRenderer: (params) => this.arrowIndicator(params), width: 250 }]
          // },
          {
            headerName: string.columnName.RASK_DOLLAR,
            headerTooltip: string.columnName.RASK_DOLLAR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_R",
                tooltipField: "CY_R_AB",
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.ASK,
            headerTooltip: string.columnName.ASK,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_A",
                tooltipField: "CY_A_AB",
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.YIELD,
            headerTooltip: string.columnName.YIELD,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Y",
                tooltipField: "CY_Y_AB",
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Y",
                tooltipField: "VLY_Y_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                hide: firstColumnName === "Ancillary",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.LOAD_FACTOR,
            headerTooltip: string.columnName.LOAD_FACTOR,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_L",
                tooltipField: "CY_L_AB",
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_L",
                tooltipField: "VLY_L_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                hide: isAncillary,
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var routeRegionTableDatas = [];
        response.data.TableData.forEach((key) => {
          routeRegionTableDatas.push({
            Ancillary_Full_Name: key.AncillaryName,
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_Avg:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_Rev:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_CY)
                : this.convertZeroValueToBlank(key.Rask_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_VLY)
                : this.convertZeroValueToBlank(key.Rask_VLY),
            CY_A: this.convertZeroValueToBlank(key.ASK_CY),
            VLY_A: this.convertZeroValueToBlank(key.ASK_VLY),
            CY_Y: this.convertZeroValueToBlank(key.yield_CY),
            VLY_Y: this.convertZeroValueToBlank(key.yield_VLY),
            CY_L: this.convertZeroValueToBlank(key.LoadFactor_CY),
            VLY_L: this.convertZeroValueToBlank(key.LoadFactor_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_Avg_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_Rev_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_CY)
                : window.numberWithCommas(key.Rask_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_VLY)
                : window.numberWithCommas(key.Rask_VLY),
            CY_A_AB: window.numberWithCommas(key.ASK_CY),
            VLY_A_AB: window.numberWithCommas(key.ASK_VLY),
            CY_Y_AB: window.numberWithCommas(key.yield_CY),
            VLY_Y_AB: window.numberWithCommas(key.yield_VLY),
            CY_L_AB: window.numberWithCommas(key.LoadFactor_CY),
            VLY_L_AB: window.numberWithCommas(key.LoadFactor_VLY),
          });
        });

        var totalData = [];
        response.data.Total.forEach((key) => {
          totalData.push({
            // 'Ancillary_Full_Name': 'Total',
            firstColumnName: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: `${this.convertZeroValueToBlank(
              key.Bookings_TKT
            )}${this.showPercent(key.Bookings_TKT)}`,
            "FRCT/Act_P": this.convertZeroValueToBlank(key.Passenger_FRCT),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            "FRCT/Act_A":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_FRCT)
                : this.convertZeroValueToBlank(key.AvgFare_FRCT),
            TGT_A: this.convertZeroValueToBlank(key.AvgFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AvgFare_VTG),
            VLY_Avg:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalAvgFare_VLY)
                : this.convertZeroValueToBlank(key.AvgFare_VLY),
            "FRCT/Act_R":
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_CY)
                : this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_Rev:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRevenue_VLY)
                : this.convertZeroValueToBlank(key.Revenue_VLY),
            Avail: this.convertZeroValueToBlank(key.Avail),
            CY_AL: this.convertZeroValueToBlank(key.AL_CY),
            VLY_AL: this.convertZeroValueToBlank(key.AL_VLY),
            CY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_CY)
                : this.convertZeroValueToBlank(key.Rask_CY),
            VLY_R:
              currency === "lc"
                ? this.convertZeroValueToBlank(key.LocalRask_VLY)
                : this.convertZeroValueToBlank(key.Rask_VLY),
            CY_A: this.convertZeroValueToBlank(key.ASK_CY),
            VLY_A: this.convertZeroValueToBlank(key.ASK_VLY),
            CY_Y: this.convertZeroValueToBlank(key.yield_CY),
            VLY_Y: this.convertZeroValueToBlank(key.yield_VLY),
            CY_L: this.convertZeroValueToBlank(key.LoadFactor_CY),
            VLY_L: this.convertZeroValueToBlank(key.LoadFactor_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            "FRCT/Act_P_AB": window.numberWithCommas(key.Passenger_FRCT),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            "FRCT/Act_A_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_FRCT)
                : window.numberWithCommas(key.AvgFare_FRCT),
            TGT_A_AB: window.numberWithCommas(key.AvgFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_Avg_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalAvgFare_VLY)
                : window.numberWithCommas(key.AvgFare_VLY),
            "FRCT/Act_R_AB":
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_CY)
                : window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_Rev_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRevenue_VLY)
                : window.numberWithCommas(key.Revenue_VLY),
            CY_AL_AB: window.numberWithCommas(key.AL_CY),
            VLY_AL_AB: window.numberWithCommas(key.AL_VLY),
            CY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_CY)
                : window.numberWithCommas(key.Rask_CY),
            VLY_R_AB:
              currency === "lc"
                ? window.numberWithCommas(key.LocalRask_VLY)
                : window.numberWithCommas(key.Rask_VLY),
            CY_A_AB: window.numberWithCommas(key.ASK_CY),
            VLY_A_AB: window.numberWithCommas(key.ASK_VLY),
            CY_Y_AB: window.numberWithCommas(key.yield_CY),
            VLY_Y_AB: window.numberWithCommas(key.yield_VLY),
            CY_L_AB: window.numberWithCommas(key.LoadFactor_CY),
            VLY_L_AB: window.numberWithCommas(key.LoadFactor_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            routeRegionTableDatas: routeRegionTableDatas,
            currentAccess: response.data.CurrentAccess,
            tabName: response.data.ColumnName,
            firstTabName: response.data.first_ColumnName,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return routeRegionTable;
  }

  getRouteCabinDetails(
    getYear,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/routecabinWiseDetails?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${flight}`;

    var cabinTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: string.columnName.RBD,
            field: "RBD",
            tooltipField: "RBD_AB",
            alignLeft: true,
          },
          {
            headerName: "Bookings",
            field: "Booking",
            tooltipField: "Booking_AB",
          },
          {
            headerName: "VLY(%)",
            field: "VLY(%)",
            tooltipField: "VLY(%)_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
          {
            headerName: "Ticketed Average Fare(SR)",
            field: "Ticketed Average Fare(SR)",
            tooltipField: "Ticketed Average Fare(SR)_AB",
          },
          {
            headerName: "VLY(%)TKT",
            field: "VLY(%)TKT",
            tooltipField: "VLY(%)TKT_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
          },
        ];

        var F = response.data.Data.filter((d) => d.Cabin === "F");
        var J = response.data.Data.filter((d) => d.Cabin === "J");
        var Y = response.data.Data.filter((d) => d.Cabin === "Y");

        var Total_F = response.data.Total.filter((d) => d.RBD === "Total of F");
        var Total_J = response.data.Total.filter((d) => d.RBD === "Total of J");
        var Total_Y = response.data.Total.filter((d) => d.RBD === "Total of Y");

        var mergedCabinData = [
          ...Total_F,
          ...F,
          ...Total_J,
          ...J,
          ...Total_Y,
          ...Y,
        ];
        var cabinData = [];

        mergedCabinData.forEach((key) => {
          cabinData.push({
            Cabin: key.Cabin,
            RBD: key.RBD,
            Booking: this.convertZeroValueToBlank(key.Bookings_CY),
            "VLY(%)": this.convertZeroValueToBlank(key.Bookings_VLY),
            "Ticketed Average Fare(SR)": this.convertZeroValueToBlank(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT": this.convertZeroValueToBlank(key.TicketedAverage_VLY),
            Booking_AB: window.numberWithCommas(key.Bookings_CY),
            "VLY(%)_AB": window.numberWithCommas(key.Bookings_VLY),
            "Ticketed Average Fare(SR)_AB": window.numberWithCommas(
              key.TicketedAverage_CY
            ),
            "VLY(%)TKT_AB": window.numberWithCommas(key.TicketedAverage_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            cabinData: cabinData,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return cabinTable;
  }

  getRouteLineCharts(
    displayName,
    group,
    region,
    country,
    route,
    leg,
    flight,
    getCabin
  ) {
    let link = "";

    if (displayName === string.columnName.BOOKINGS) {
      link = "routebooking";
    }
    if (displayName === string.columnName.PASSENGER_OD) {
      link = "routepassenger";
    }
    if (displayName === string.columnName.AVERAGE_FARE_$) {
      link = "routeavgfare";
    }
    if (displayName === string.columnName.REVENUE_$) {
      link = "routerevenue";
    }
    const url = `${API_URL}/${link}?selectedRouteGroup=${group}&selectedRouteRegion=${region}&selectedRouteCountry=${country}&selectedRoute=${route}&selectedLeg=${leg}&flight=${flight}&getCabinValue=${getCabin}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  getRouteLineChartsForecast(
    displayName,
    group,
    region,
    country,
    route,
    leg,
    flight,
    getCabin,
    gettingYear,
    gettingMonth
  ) {
    let link = "";

    if (displayName === string.columnName.BOOKINGS) {
      link = "routebooking";
    }
    if (displayName === "Passenger Forecast") {
      link = "routePassengerForeGraph";
    }
    if (displayName === "Average fare Forecast") {
      link = "routeAvgFareForeGraph";
    }
    if (displayName === "Revenue Forecast") {
      link = "routeRevenueForeGraph";
    }
    const url = `${API_URL}/${link}?getYear=${gettingYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${group}&selectedRouteRegion=${region}&selectedRouteCountry=${country}&selectedRoute=${route}&selectedLeg=${leg}&flight=${flight}&getCabinValue=${getCabin}`;
    return axios
      .get(url, this.getDefaultHeader())
      .then((response) => response.data.response)
      .catch((error) => {
        console.log(error);
      });
  }

  exportCSVRouteMonthlyURL(
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue
  ) {
    const url = `routeDataMonthly?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${flight}`;
    return url;
  }

  exportCSVRouteDrillDownURL(
    getYear,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue,
    type
  ) {
    const url = `routeDataDrillDown?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${flight}&type=${type}`;
    return url;
  }

  getPOSContributionData(
    getYear,
    currency,
    gettingMonth,
    regionId,
    countryId,
    routeId,
    getCabinValue,
    posContributionTable,
    count,
    leg,
    flight
  ) {
    const url = `${API_URL}/poscontributionnew?getYear=${getYear}&gettingMonth=${gettingMonth}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&tableType=${posContributionTable}&page_num=${count}`;
    var posContri = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        localStorage.setItem("posContributionDownloadURL", url);

        var columnNameOD = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.ROUTE,
                headerTooltip: string.columnName.ROUTE,
                field: "Route",
                tooltipField: "Route",
                width: 300,
                alignLeft: true,
              },
            ],
          },
          {
            headerName: string.columnName.POS,
            headerTooltip: string.columnName.POS,
            field: "POS",
            tooltipField: "POS",
            width: 300,
            alignLeft: true,
          },
          {
            headerName: "",
            children: [
              {
                headerName: "Leg",
                headerTooltip: "Leg",
                field: "Leg",
                tooltipField: "Leg",
                width: 300,
                alignLeft: true,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.FLIGHT,
                headerTooltip: string.columnName.FLIGHT,
                field: "Flight",
                tooltipField: "Flight",
                width: 300,
                alignLeft: true,
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                headerTooltip: string.columnName.OD,
                field: "OD",
                tooltipField: "OD",
                width: 300,
                alignLeft: true,
                hide: posContributionTable == "Segment",
              },
            ],
          },
          {
            headerName: "",
            children: [
              {
                headerName: "Segment",
                headerTooltip: "Segment",
                field: "Segment",
                tooltipField: "Segment",
                width: 300,
                alignLeft: true,
                hide: posContributionTable == "OD",
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER_OD,
            headerTooltip: string.columnName.PASSENGER_OD,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Passenger",
                tooltipField: "CY_Passenger_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
                hide: posContributionTable == "Segment",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            headerTooltip: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "AvgFare_CY",
                tooltipField: "AvgFare_CY_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
                hide: posContributionTable == "Segment",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerTooltip: string.columnName.REVENUE_$,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_Revenue",
                tooltipField: "CY_Revenue_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
                hide: posContributionTable == "Segment",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_Rev",
                tooltipField: "VLY_Rev_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowDataOD = [];
        let that = this;
        let resp = {};
        if (response.data.response) {
          resp = response.data.response;
        } else {
          resp = response.data;
        }
        resp.TableData.forEach(function (key) {
          rowDataOD.push({
            Route: key.Route,
            POS: key.POS,
            Leg: key.Leg,
            Flight: key.FlightNumber,
            OD: key.OD,
            Segment: key.Segment,
            CY_Passenger: that.convertZeroValueToBlank(key.CY_Passenger),
            CY_Passenger_AB: window.numberWithCommas(key.CY_Passenger),
            VTG_P: that.convertZeroValueToBlank(key.Passenger_VTG),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P: that.convertZeroValueToBlank(key.Passenger_VLY),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            CY_Revenue: that.convertZeroValueToBlank(key.CY_Revenue),
            CY_Revenue_AB: window.numberWithCommas(key.CY_Revenue),
            VTG_R: that.convertZeroValueToBlank(key.Revenue_VTG),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_Rev: that.convertZeroValueToBlank(key.Revenue_VLY),
            VLY_Rev_AB: window.numberWithCommas(key.Revenue_VLY),
            AvgFare_CY: that.convertZeroValueToBlank(key.AvgFare_CY),
            AvgFare_CY_AB: window.numberWithCommas(key.AvgFare_CY),
            VTG_A: that.convertZeroValueToBlank(key.AvgFare_VTG),
            VTG_A_AB: window.numberWithCommas(key.AvgFare_VTG),
            VLY_A: that.convertZeroValueToBlank(key.AvgFare_VLY),
            VLY_A_AB: window.numberWithCommas(key.AvgFare_VLY),
          });
        });

        return [
          {
            columnNameOD: columnNameOD,
            rowDataOD: rowDataOD,
            currentPage: resp.pageNumber,
            totalPages: resp.totalPages,
            totalRecords: resp.totalRecords,
            paginationSize: resp.paginationLimit,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return posContri;
  }

  //Forecast Accuracy API
  getForecastMonthTable(
    startDate,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue
  ) {
    const url = `${API_URL}/forecastaccuracymonthly?startdate=${startDate}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(commonOD)}`;

    var forecastMonthTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_p",
                tooltipField: "target_p_AB",
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_p",
                tooltipField: "forecast_p_AB",
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_p",
                tooltipField: "actual_p_AB",
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_p",
                tooltipField: "error_p_AB",
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_a",
                tooltipField: "target_a_AB",
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_a",
                tooltipField: "forecast_a_AB",
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_a",
                tooltipField: "actual_a_AB",
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_a",
                tooltipField: "error_a_AB",
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_r",
                tooltipField: "target_r_AB",
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_r",
                tooltipField: "forecast_r_AB",
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_r",
                tooltipField: "actual_r_AB",
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_r",
                tooltipField: "error_r_AB",
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
        ];

        var rowData = [];

        var responseData = response.data.response;

        responseData.TableData.forEach((key) => {
          rowData.push({
            Month: key.MonthName === null ? "---" : key.MonthName,
            target_p: this.convertZeroValueToBlank(key.Target_pax),
            forecast_p: this.convertZeroValueToBlank(key.Forecast_pax),
            actual_p: this.convertZeroValueToBlank(key.Actual_Pax),
            error_p: this.convertZeroValueToBlank(key.Error_Pax),
            target_a: this.convertZeroValueToBlank(key.target_avg),
            forecast_a: this.convertZeroValueToBlank(key.Forecast_avg),
            actual_a: this.convertZeroValueToBlank(key.Actual_avg),
            error_a: this.convertZeroValueToBlank(key.Error_Avg),
            target_r: this.convertZeroValueToBlank(key.Target_Rev),
            forecast_r: this.convertZeroValueToBlank(key.Forecast_Rev),
            actual_r: this.convertZeroValueToBlank(key.Actual_Rev),
            error_r: this.convertZeroValueToBlank(key.Error_Rev),
            target_p_AB: window.numberWithCommas(key.Target_pax),
            forecast_p_AB: window.numberWithCommas(key.Forecast_pax),
            actual_p_AB: window.numberWithCommas(key.Actual_Pax),
            error_p_AB: window.numberWithCommas(key.Error_Pax),
            target_a_AB: window.numberWithCommas(key.target_avg),
            forecast_a_AB: window.numberWithCommas(key.Forecast_avg),
            actual_a_AB: window.numberWithCommas(key.Actual_avg),
            error_a_AB: window.numberWithCommas(key.Error_Avg),
            target_r_AB: window.numberWithCommas(key.Target_Rev),
            forecast_r_AB: window.numberWithCommas(key.Forecast_Rev),
            actual_r_AB: window.numberWithCommas(key.Actual_Rev),
            error_r_AB: window.numberWithCommas(key.Error_Rev),
            Year: key.Year,
            MonthName: key.monthfullname,
          });
        });

        var totalData = [];
        responseData.Total.forEach((key) => {
          totalData.push({
            Month: "Total",
            target_p: this.convertZeroValueToBlank(key.Target_pax),
            forecast_p: this.convertZeroValueToBlank(key.Forecast_pax),
            actual_p: this.convertZeroValueToBlank(key.Actual_Pax),
            error_p: this.convertZeroValueToBlank(key.Error_Pax),
            target_a: this.convertZeroValueToBlank(key.target_avg),
            forecast_a: this.convertZeroValueToBlank(key.Forecast_avg),
            actual_a: this.convertZeroValueToBlank(key.Actual_avg),
            error_a: this.convertZeroValueToBlank(key.Error_Avg),
            target_r: this.convertZeroValueToBlank(key.Target_Rev),
            forecast_r: this.convertZeroValueToBlank(key.Forecast_Rev),
            actual_r: this.convertZeroValueToBlank(key.Actual_Rev),
            error_r: this.convertZeroValueToBlank(key.Error_Rev),
            target_p_AB: window.numberWithCommas(key.Target_pax),
            forecast_p_AB: window.numberWithCommas(key.Forecast_pax),
            actual_p_AB: window.numberWithCommas(key.Actual_Pax),
            error_p_AB: window.numberWithCommas(key.Error_Pax),
            target_a_AB: window.numberWithCommas(key.target_avg),
            forecast_a_AB: window.numberWithCommas(key.Forecast_avg),
            actual_a_AB: window.numberWithCommas(key.Actual_avg),
            error_a_AB: window.numberWithCommas(key.Error_Avg),
            target_r_AB: window.numberWithCommas(key.Target_Rev),
            forecast_r_AB: window.numberWithCommas(key.Forecast_Rev),
            actual_r_AB: window.numberWithCommas(key.Actual_Rev),
            error_r_AB: window.numberWithCommas(key.Error_Rev),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentAccess: responseData.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log(error);
      });

    return forecastMonthTable;
  }

  getForecastDrillDownData(
    getYear,
    startDate,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    type
  ) {
    let currentMonth = new Date().getMonth() + 1;
    const url = `${API_URL}/forecastaccuracydrilldown?startdate=${startDate}&getYear=${getYear}&gettingMonth=${gettingMonth == currentMonth ? gettingMonth - 1 : gettingMonth
      }&${Params(
        regionId,
        countryId,
        cityId,
        getCabinValue
      )}&commonOD=${encodeURIComponent(commonOD)}&type=${type}`;
    var forecastRegionTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const firstColumnName = response.data.response.ColumnName;
        const currentMonth = new Date().getMonth() + 1;
        const date = new Date();
        const currmonthdate = new Date(date.getFullYear(), date.getMonth(), 2)
          .toISOString()
          .slice(0, 10);
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                width: 250,
                alignLeft: true,
                underline:
                  type === "Null" &&
                    firstColumnName !== "Cabin" &&
                    firstColumnName !== "POS" &&
                    startDate !== currmonthdate
                    ? true
                    : false,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_p",
                tooltipField: "target_p_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_p",
                tooltipField: "forecast_p_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_p",
                tooltipField: "actual_p_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_p",
                tooltipField: "error_p_AB",
                sortable: true,
                comparator: this.customSorting,
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_a",
                tooltipField: "target_a_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_a",
                tooltipField: "forecast_a_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_a",
                tooltipField: "actual_a_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_a",
                tooltipField: "error_a_AB",
                sortable: true,
                comparator: this.customSorting,
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE,
            children: [
              {
                headerName: string.columnName.TARGET,
                field: "target_r",
                tooltipField: "target_r_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.Forecast,
                field: "forecast_r",
                tooltipField: "forecast_r_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACTUAL,
                field: "actual_r",
                tooltipField: "actual_r_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.ACCURACY,
                field: "error_r",
                tooltipField: "error_r_AB",
                sortable: true,
                comparator: this.customSorting,
                cellRenderer: (params) => this.accuracyArrowIndicator(params),
              },
            ],
          },
        ];

        var rowData = [];
        var responseData = response.data.response;
        responseData.TableData.forEach((key) => {
          rowData.push({
            firstColumnName: key.ColumnName === null ? "---" : key.ColumnName,
            target_p: this.convertZeroValueToBlank(key.Target_pax),
            forecast_p: this.convertZeroValueToBlank(key.Forecast_pax),
            actual_p: this.convertZeroValueToBlank(key.Actual_Pax),
            error_p: this.convertZeroValueToBlank(key.Error_Pax),
            target_a: this.convertZeroValueToBlank(key.target_avg),
            forecast_a: this.convertZeroValueToBlank(key.Forecast_avg),
            actual_a: this.convertZeroValueToBlank(key.Actual_avg),
            error_a: this.convertZeroValueToBlank(key.Error_Avg),
            target_r: this.convertZeroValueToBlank(key.Target_Rev),
            forecast_r: this.convertZeroValueToBlank(key.Forecast_Rev),
            actual_r: this.convertZeroValueToBlank(key.Actual_Rev),
            error_r: this.convertZeroValueToBlank(key.Error_Rev),
            target_p_AB: window.numberWithCommas(key.Target_pax),
            forecast_p_AB: window.numberWithCommas(key.Forecast_pax),
            actual_p_AB: window.numberWithCommas(key.Actual_Pax),
            error_p_AB: window.numberWithCommas(key.Error_Pax),
            target_a_AB: window.numberWithCommas(key.target_avg),
            forecast_a_AB: window.numberWithCommas(key.Forecast_avg),
            actual_a_AB: window.numberWithCommas(key.Actual_avg),
            error_a_AB: window.numberWithCommas(key.Error_Avg),
            target_r_AB: window.numberWithCommas(key.Target_Rev),
            forecast_r_AB: window.numberWithCommas(key.Forecast_Rev),
            actual_r_AB: window.numberWithCommas(key.Actual_Rev),
            error_r_AB: window.numberWithCommas(key.Error_Rev),
          });
        });

        var totalData = [];

        responseData.Total.forEach((key) => {
          totalData.push({
            Ancillary_Full_Name: "Total",
            firstColumnName: "Total",
            target_p: this.convertZeroValueToBlank(key.Target_pax),
            forecast_p: this.convertZeroValueToBlank(key.Forecast_pax),
            actual_p: this.convertZeroValueToBlank(key.Actual_Pax),
            error_p: this.convertZeroValueToBlank(key.Error_Pax),
            target_a: this.convertZeroValueToBlank(key.target_avg),
            forecast_a: this.convertZeroValueToBlank(key.Forecast_avg),
            actual_a: this.convertZeroValueToBlank(key.Actual_avg),
            error_a: this.convertZeroValueToBlank(key.Error_Avg),
            target_r: this.convertZeroValueToBlank(key.Target_Rev),
            forecast_r: this.convertZeroValueToBlank(key.Forecast_Rev),
            actual_r: this.convertZeroValueToBlank(key.Actual_Rev),
            error_r: this.convertZeroValueToBlank(key.Error_Rev),
            target_p_AB: window.numberWithCommas(key.Target_pax),
            forecast_p_AB: window.numberWithCommas(key.Forecast_pax),
            actual_p_AB: window.numberWithCommas(key.Actual_Pax),
            error_p_AB: window.numberWithCommas(key.Error_Pax),
            target_a_AB: window.numberWithCommas(key.target_avg),
            forecast_a_AB: window.numberWithCommas(key.Forecast_avg),
            actual_a_AB: window.numberWithCommas(key.Actual_avg),
            error_a_AB: window.numberWithCommas(key.Error_Avg),
            target_r_AB: window.numberWithCommas(key.Target_Rev),
            forecast_r_AB: window.numberWithCommas(key.Forecast_Rev),
            actual_r_AB: window.numberWithCommas(key.Actual_Rev),
            error_r_AB: window.numberWithCommas(key.Error_Rev),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            currentAccess: responseData.CurrentAccess,
            totalData: totalData,
            tabName: responseData.ColumnName,
            firstTabName: responseData.first_ColumnName,
          },
        ];
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return forecastRegionTable;
  }

  //Top Markets API
  getTopMarkets(
    page_num,
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getOD,
    getLeftTableValue
  ) {
    const url = `${API_URL}/topmarkettable?page_num=${page_num}&startDate=${startDate}&endDate=${endDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getOD=${String.addQuotesforMultiSelect(
      getOD
    )}&getLeftTableValue=${encodeURIComponent(getLeftTableValue)}`;
    const downloadurl = `${API_URL}/FullYearDownloadTopMarket?startDate=${startDate}&endDate=${endDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getOD=${String.addQuotesforMultiSelect(getOD)}`;

    localStorage.setItem("topMarketDownloadURL", downloadurl);

    var topMarketTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let forecastzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let forecastTGTVisible =
          forecastzeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_VTG === 0 || d.Revenue_VTG === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                field: string.columnName.OD,
                alignLeft: true,
                underline: true,
              },
            ],
            cellStyle: { color: "red" },
          },
          {
            headerName: string.columnName.MIDT_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDT",
                tooltipField: "CY_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MIDT",
                tooltipField: "LY_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDT",
                tooltipField: "VLY(%)_MIDT_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_MIDT",
                tooltipField: "YTD_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_AIRLINES_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDTA",
                tooltipField: "CY_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MIDTA",
                tooltipField: "LY_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDTA",
                tooltipField: "VLY(%)_MIDTA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.O_AND_D_PASSENGER_FORECAST,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_OD",
                tooltipField: "Forecast_OD_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: "TGT",
                field: "TGT_OD",
                tooltipField: "TGT_OD_AB",
                hide: forecastTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: "VTG(%)",
                field: "VTG(%)_OD",
                tooltipField: "VTG(%)_OD_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: forecastTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_OD",
                tooltipField: "VLY(%)_OD_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE,
            hide: revenueTGTVisible,
            children: [
              {
                headerName: "VTG(%)",
                field: "VTG(%)_REV",
                tooltipField: "VTG(%)_REV_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_AL",
                tooltipField: "CY_AL_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_AL",
                tooltipField: "VLY(%)_AL_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: "YTD(Abs)",
                field: "YTD_AL",
                tooltipField: "YTD_AL_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: string.columnName.TOP_COMPETITOR, field: 'TOP Competitor %', width: 300 }]
          // }
        ];

        var rowData = [];
        var responseData = response.data.response[0];
        responseData.TableData.forEach((key) => {
          rowData.push({
            OD: key.OD,
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            LY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_LY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            Forecast_OD: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_OD: this.convertZeroValueToBlank(key.Passenger_TGT),
            "VTG(%)_OD": this.convertZeroValueToBlank(key.Passenger_VTG),
            "VLY(%)_OD": this.convertZeroValueToBlank(key.Passenger_VLY),
            "VTG(%)_REV": this.convertZeroValueToBlank(key.Revenue_VTG),
            CY_AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_AL": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_AL: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            "TOP Competitor %": "---",
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            LY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_LY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.MS_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            Forecast_OD_AB: window.numberWithCommas(key.Forecast),
            TGT_OD_AB: window.numberWithCommas(key.Passenger_TGT),
            "VTG(%)_OD_AB": window.numberWithCommas(key.Forecast_VTG),
            "VLY(%)_OD_AB": window.numberWithCommas(key.Forecast_VLY),
            "VTG(%)_REV_AB": window.numberWithCommas(key.Revenue_VTG),
            CY_AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_AL_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_AL_AB: window.numberWithCommas(key.MS_Airlines_YTD),
          });
        });

        var totalData = [];
        responseData.Total.forEach((key) => {
          totalData.push({
            OD: "Total",
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            LY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_LY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            Forecast_OD: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_OD: this.convertZeroValueToBlank(key.Passenger_TGT),
            "VTG(%)_OD": this.convertZeroValueToBlank(key.Passenger_VTG),
            "VLY(%)_OD": this.convertZeroValueToBlank(key.Passenger_VLY),
            "VTG(%)_REV": this.convertZeroValueToBlank(key.Revenue_VTG),
            CY_AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_AL": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_AL: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            "TOP Competitor %": "---",
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            LY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_LY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.MS_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            Forecast_OD_AB: window.numberWithCommas(key.Forecast),
            TGT_OD_AB: window.numberWithCommas(key.Passenger_TGT),
            "VTG(%)_OD_AB": window.numberWithCommas(key.Forecast_VTG),
            "VLY(%)_OD_AB": window.numberWithCommas(key.Forecast_VLY),
            "VTG(%)_REV_AB": window.numberWithCommas(key.Revenue_VTG),
            CY_AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_AL_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_AL_AB: window.numberWithCommas(key.MS_Airlines_YTD),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            currentPage: responseData.pageNumber,
            totalPages: responseData.totalPages,
            totalRecords: responseData.totalRecords,
            paginationSize: responseData.paginationLimit,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topMarketTable;
  }

  getTopCompetitors(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getOD,
    getLeftTableValue
  ) {
    const url = `${API_URL}/topOdcompetitors?startDate=${startDate}&endDate=${endDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getOD=${getOD}&getLeftTableValue=${encodeURIComponent(
      getLeftTableValue
    )}`;

    var topCompetitorsTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.AIRLINE,
                field: "Airline",
                tooltipField: "Airline",
                alignLeft: true,
                underline: true,
              },
            ],
          },
          // {
          //     headerName: string.columnName.MIDT_BOOKED_PASSENGER,
          //     children: [{ headerName: string.columnName.CY, field: 'CY_MIDT' }, { headerName: string.columnName.VLY, field: 'VLY(%)_MIDT', cellRenderer: (params) => this.arrowIndicator(params) }, { headerName: string.columnName.YTD, field: 'YTD_MIDT' }]
          // },
          {
            headerName: string.columnName.MIDT_AIRLINES_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDTA",
                tooltipField: "CY_MIDTA_AB",
                underline: getLeftTableValue === "Null" ? true : false,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MIDTA",
                tooltipField: "LY_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDTA",
                tooltipField: "VLY(%)_MIDTA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_AL",
                tooltipField: "CY_AL_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_AL",
                tooltipField: "VLY(%)_AL_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_AL",
                tooltipField: "YTD_AL_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            Airline: key.airlines,
            // 'CY_MIDT': this.convertZeroValueToBlank(key.BookedPassenger_CY),
            // 'VLY(%)_MIDT': this.convertZeroValueToBlank(key.BookedPassenger_VLY),
            // 'YTD_MIDT': this.convertZeroValueToBlank(key.MIDT_Booked_Airlines_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_AL": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_AL: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_AL_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_AL_AB: window.numberWithCommas(key.MS_Airlines_YTD),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Airline: "Total",
            // 'CY_MIDT': this.convertZeroValueToBlank(key.BookedPassenger_CY),
            // 'VLY(%)_MIDT': this.convertZeroValueToBlank(key.BookedPassenger_VLY),
            // 'YTD_MIDT': this.convertZeroValueToBlank(key.MIDT_Booked_Airlines_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_AL": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_AL: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            LY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_LY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_AL_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_AL_AB: window.numberWithCommas(key.MS_Airlines_YTD),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topCompetitorsTable;
  }

  getTopAgents(
    startDate,
    endDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getOD,
    getLeftTableValue
  ) {
    const url = `${API_URL}/topODagents?startDate=${startDate}&endDate=${endDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getOD=${getOD}&getLeftTableValue=${encodeURIComponent(
      getLeftTableValue
    )}`;

    var topAgentsTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.NAME,
                field: "Name",
                tooltipField: "Name",
                width: 300,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDT",
                tooltipField: "CY_MIDT_AB",
                underline: getLeftTableValue === "Null" ? true : false,
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDT",
                tooltipField: "VLY(%)_MIDT_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_MIDT",
                tooltipField: "YTD_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_AIRLINES_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDTA",
                tooltipField: "CY_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDTA",
                tooltipField: "VLY(%)_MIDTA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_MIDTA",
                tooltipField: "YTD_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: string.columnName.AL_MARKET_SHARE,
          //     children: [{ headerName: string.columnName.CY_ABS, field: 'CY_ABS', width: 250 }]
          // },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            Name: key.AgentName,
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            YTD_MIDTA: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            CY_ABS: this.convertZeroValueToBlank(key.MarketShare_CY),
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.MS_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            YTD_MIDTA_AB: window.numberWithCommas(key.MS_Airlines_YTD),
            CY_ABS_AB: window.numberWithCommas(key.MarketShare_CY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Name: "Total",
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_CY
            ),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            YTD_MIDTA: this.convertZeroValueToBlank(key.MS_Airlines_YTD),
            CY_ABS: this.convertZeroValueToBlank(key.MarketShare_CY),
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.MS_YTD),
            CY_MIDTA_AB: window.numberWithCommas(
              key.AirlinesBookedPassenger_CY
            ),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(
              key.AirlinesBookedPassenger_VLY
            ),
            YTD_MIDTA_AB: window.numberWithCommas(key.MS_Airlines_YTD),
            CY_ABS_AB: window.numberWithCommas(key.MarketShare_CY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topAgentsTable;
  }

  //Competitor Analysis
  getCompetitorAnalysis(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getAirline
  ) {
    const url = `${API_URL}/competitorcabin?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getAirline=${encodeURIComponent(getAirline)}`;

    var topMarketCabin = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.CABIN,
                field: "Cabin",
                alignLeft: true,
              },
            ],
          },
          {
            headerName: getAirline,
            children: [
              {
                headerName: string.columnName.MIDT_CY_BOOKINGS,
                field: "MIDT CY Bookings_C",
                tooltipField: "MIDT CY Bookings_C_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.MIDT_LY_BOOKINGS,
                field: "MIDT LY Bookings_C",
                tooltipField: "MIDT LY Bookings_C_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.MIDT_BOOKINGS_VLY,
                field: "MIDT Bookings VLY(%)_C",
                tooltipField: "MIDT Bookings VLY(%)_C_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.CY_MS,
                field: "CY MS_C",
                tooltipField: "CY MS_C_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY_MS,
                field: "LY MS_C",
                tooltipField: "LY MS_C_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AL,
            children: [
              {
                headerName: string.columnName.MIDT_CY_BOOKINGS,
                field: "MIDT CY Bookings_A",
                tooltipField: "MIDT CY Bookings_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.MIDT_LY_BOOKINGS,
                field: "MIDT LY Bookings_A",
                tooltipField: "MIDT LY Bookings_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.MIDT_BOOKINGS_VLY,
                field: "MIDT Bookings VLY(%)_A",
                tooltipField: "MIDT Bookings VLY(%)_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                width: 250,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.CY_MS,
                field: "CY MS_A",
                tooltipField: "CY MS_A_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.LY_MS,
                field: "LY MS_A",
                tooltipField: "LY MS_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];

        response.data.response[0].TableData.forEach((A) => {
          rowData.push({
            Cabin: A.CX_Cabin,
            "MIDT CY Bookings_C": this.convertZeroValueToBlank(A.CX_MIDT_cy),
            "MIDT LY Bookings_C": this.convertZeroValueToBlank(A.CX_MIDT_ly),
            "MIDT Bookings VLY(%)_C": this.convertZeroValueToBlank(A.CX_VLY),
            "CY MS_C": this.convertZeroValueToBlank(A.CX_CY_MS),
            "LY MS_C": this.convertZeroValueToBlank(A.CX_LY_MS),
            "MIDT CY Bookings_A": this.convertZeroValueToBlank(A.AL_MIDT_cy),
            "MIDT LY Bookings_A": this.convertZeroValueToBlank(A.AL_MIDT_ly),
            "MIDT Bookings VLY(%)_A": this.convertZeroValueToBlank(A.AL_VLY),
            "CY MS_A": this.convertZeroValueToBlank(A.AL_CY_MS),
            "LY MS_A": this.convertZeroValueToBlank(A.AL_LY_MS),
            "MIDT CY Bookings_C_AB": window.numberWithCommas(A.CX_MIDT_cy),
            "MIDT LY Bookings_C_AB": window.numberWithCommas(A.CX_MIDT_ly),
            "MIDT Bookings VLY(%)_C_AB": window.numberWithCommas(A.CX_VLY),
            "CY MS_C_AB": window.numberWithCommas(A.CX_CY_MS),
            "LY MS_C_AB": window.numberWithCommas(A.CX_LY_MS),
            "MIDT CY Bookings_A_AB": window.numberWithCommas(A.AL_MIDT_cy),
            "MIDT LY Bookings_A_AB": window.numberWithCommas(A.AL_MIDT_ly),
            "MIDT Bookings VLY(%)_A_AB": window.numberWithCommas(A.AL_VLY),
            "CY MS_A_AB": window.numberWithCommas(A.AL_CY_MS),
            "LY MS_A_AB": window.numberWithCommas(A.AL_LY_MS),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((A) => {
          totalData.push({
            Cabin: "Total",
            "MIDT CY Bookings_C": this.convertZeroValueToBlank(A.CX_MIDT_cy),
            "MIDT LY Bookings_C": this.convertZeroValueToBlank(A.CX_MIDT_ly),
            "MIDT Bookings VLY(%)_C": this.convertZeroValueToBlank(A.CX_VLY),
            "CY MS_C": this.convertZeroValueToBlank(A.CX_CY_MS),
            "LY MS_C": this.convertZeroValueToBlank(A.CX_LY_MS),
            "MIDT CY Bookings_A": this.convertZeroValueToBlank(A.AL_MIDT_cy),
            "MIDT LY Bookings_A": this.convertZeroValueToBlank(A.AL_MIDT_ly),
            "MIDT Bookings VLY(%)_A": this.convertZeroValueToBlank(A.AL_VLY),
            "CY MS_A": this.convertZeroValueToBlank(A.AL_CY_MS),
            "LY MS_A": this.convertZeroValueToBlank(A.AL_LY_MS),
            "MIDT CY Bookings_C_AB": window.numberWithCommas(A.CX_MIDT_cy),
            "MIDT LY Bookings_C_AB": window.numberWithCommas(A.CX_MIDT_ly),
            "MIDT Bookings VLY(%)_C_AB": window.numberWithCommas(A.CX_VLY),
            "CY MS_C_AB": window.numberWithCommas(A.CX_CY_MS),
            "LY MS_C_AB": window.numberWithCommas(A.CX_LY_MS),
            "MIDT CY Bookings_A_AB": window.numberWithCommas(A.AL_MIDT_cy),
            "MIDT LY Bookings_A_AB": window.numberWithCommas(A.AL_MIDT_ly),
            "MIDT Bookings VLY(%)_A_AB": window.numberWithCommas(A.AL_VLY),
            "CY MS_A_AB": window.numberWithCommas(A.AL_CY_MS),
            "LY MS_A_AB": window.numberWithCommas(A.AL_LY_MS),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
            // 'totalData': totalData
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topMarketCabin;
  }

  getTopMarketsForCompetitors(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getAirline,
    getTopMarkets
  ) {
    const url = `${API_URL}/competitortopmarkets?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getAirline=${encodeURIComponent(
      getAirline
    )}&getTopMarkets=${getTopMarkets}`;

    var topMarketTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                field: "OD",
                tooltipField: "OD",
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_M",
                tooltipField: "LY_M_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MS",
                tooltipField: "CY_MS_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MS",
                tooltipField: "LY_MS_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: '',
          //     children: [{ headerName: string.columnName.AIRLINE_RANK, field: 'Airline Rank', width: 300 }]
          // }
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            OD: key.OD,
            // 'MIDT': 'CX',
            CY_M: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_CY),
            LY_M: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_LY),
            VLY_M: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_MS: this.convertZeroValueToBlank(key.MarketShare_CY),
            LY_MS: this.convertZeroValueToBlank(key.MarketShare_LY),
            CY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_CY),
            LY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_LY),
            VLY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_VLY),
            CY_MS_AB: window.numberWithCommas(key.MarketShare_CY),
            LY_MS_AB: window.numberWithCommas(key.MarketShare_LY),
            "Airline Rank": "---",
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            OD: "Total",
            // 'MIDT': 'CX',
            CY_M: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_CY),
            LY_M: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_LY),
            VLY_M: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_MS: this.convertZeroValueToBlank(key.MarketShare_CY),
            LY_MS: this.convertZeroValueToBlank(key.MarketShare_LY),
            CY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_CY),
            LY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_LY),
            VLY_M_AB: window.numberWithCommas(key.AirlinesBookedPassenger_VLY),
            CY_MS_AB: window.numberWithCommas(key.MarketShare_CY),
            LY_MS_AB: window.numberWithCommas(key.MarketShare_LY),
            "Airline Rank": "---",
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topMarketTable;
  }

  getTopAgentsForCompetitors(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getAirline,
    getTopMarkets,
    getOD
  ) {
    const url = `${API_URL}/competitortopagents?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getAirline=${encodeURIComponent(
      getAirline
    )}&getTopMarkets=${getTopMarkets}&getOD=${getOD}`;

    var topAgentsTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.NAME,
                field: "Name",
                tooltipField: "Name",
                width: 300,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDT",
                tooltipField: "CY_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDT",
                tooltipField: "VLY(%)_MIDT_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_MIDT",
                tooltipField: "YTD_MIDT_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MIDTA",
                tooltipField: "CY_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY(%)_MIDTA",
                tooltipField: "VLY(%)_MIDTA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YTD,
                field: "YTD_MIDTA",
                tooltipField: "YTD_MIDTA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            Name: key.AgentName,
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.Booked_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_MIDTA: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.Booked_YTD),
            CY_MIDTA_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_MIDTA_AB: window.numberWithCommas(key.MS_YTD),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Name: "Total",
            CY_MIDT: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            "VLY(%)_MIDT": this.convertZeroValueToBlank(
              key.BookedPassenger_VLY
            ),
            YTD_MIDT: this.convertZeroValueToBlank(key.Booked_YTD),
            CY_MIDTA: this.convertZeroValueToBlank(key.MarketShare_CY),
            "VLY(%)_MIDTA": this.convertZeroValueToBlank(key.MarketShare_VLY),
            YTD_MIDTA: this.convertZeroValueToBlank(key.MS_YTD),
            CY_MIDT_AB: window.numberWithCommas(key.BookedPassenger_CY),
            "VLY(%)_MIDT_AB": window.numberWithCommas(key.BookedPassenger_VLY),
            YTD_MIDT_AB: window.numberWithCommas(key.Booked_YTD),
            CY_MIDTA_AB: window.numberWithCommas(key.MarketShare_CY),
            "VLY(%)_MIDTA_AB": window.numberWithCommas(key.MarketShare_VLY),
            YTD_MIDTA_AB: window.numberWithCommas(key.MS_YTD),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topAgentsTable;
  }

  //Agent Analysis
  getAgentAnalysis(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    agentname
  ) {
    const url = `${API_URL}/agentanalysisreport?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&agentname=${encodeURIComponent(agentname)}`;

    var agentAnalysisTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.AGENT_NAME,
                field: "AgentName",
                tooltipField: "AgentName",
                alignLeft: true,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_M",
                tooltipField: "LY_M_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_AIRLINES_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MA",
                tooltipField: "CY_MA_AB",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MA",
                tooltipField: "LY_MA_AB",
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_MA",
                tooltipField: "VLY_MA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MS",
                tooltipField: "CY_MS_AB",
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_MS",
                tooltipField: "YOY_MS_AB",
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.AL_PERCENTAGE,
                field: "AL",
                tooltipField: "AL_AB",
              },
              {
                headerName: string.columnName.YOY_PTS,
                field: "YOY(pts)",
                tooltipField: "YOY(pts)_AB",
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            AgentName: key.AgentName,
            CY_M: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            LY_M: this.convertZeroValueToBlank(key.BookedPassenger_LY),
            VLY_M: this.convertZeroValueToBlank(key.BookedPassenger_VLY),
            CY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_CY),
            LY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_LY),
            VLY_MA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_MS: this.convertZeroValueToBlank(key.Agent_MS_CY),
            YOY_MS: this.convertZeroValueToBlank(key.Agent_MS_YOY),
            AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "YOY(pts)": this.convertZeroValueToBlank(key.MarketShare_VLY),
            CY_M_AB: window.numberWithCommas(key.BookedPassenger_CY),
            LY_M_AB: window.numberWithCommas(key.BookedPassenger_LY),
            VLY_M_AB: window.numberWithCommas(key.BookedPassenger_VLY),
            CY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_CY),
            LY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_LY),
            VLY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_VLY),
            CY_MS_AB: window.numberWithCommas(key.Agent_MS_CY),
            YOY_MS_AB: window.numberWithCommas(key.Agent_MS_YOY),
            AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "YOY(pts)_AB": window.numberWithCommas(key.MarketShare_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return agentAnalysisTable;
  }

  getTopMarketsForAgent(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    getTopMarkets,
    agentname
  ) {
    const url = `${API_URL}/agenttopmarkets?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&getTopMarkets=${getTopMarkets}&agentname=${encodeURIComponent(
      agentname
    )}`;

    var topMarketTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                field: "OD",
                alignLeft: true,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_M",
                tooltipField: "CY_M_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.LY,
                field: "LY_M",
                tooltipField: "LY_M_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_M",
                tooltipField: "VLY_M_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MIDT_AIRLINES_BOOKED_PASSENGER,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MA",
                tooltipField: "CY_MA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.LY,
                field: "LY_MA",
                tooltipField: "LY_MA_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_MA",
                tooltipField: "VLY_MA_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_MS",
                tooltipField: "CY_MS_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PERCENTAGE,
                field: "YOY_MS",
                tooltipField: "YOY_MS_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AL_MARKET_SHARE,
            children: [
              {
                headerName: string.columnName.AL_PERCENTAGE,
                field: "AL",
                tooltipField: "AL_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.YOY_PTS,
                field: "YOY(pts)",
                tooltipField: "YOY(pts)_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];
        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            OD: key.OD,
            CY_M: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            LY_M: this.convertZeroValueToBlank(key.BookedPassenger_LY),
            VLY_M: this.convertZeroValueToBlank(key.BookedPassenger_VLY),
            CY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_CY),
            LY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_LY),
            VLY_MA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_MS: this.convertZeroValueToBlank(key.Agent_MS_CY),
            YOY_MS: this.convertZeroValueToBlank(key.Agent_MS_YOY),
            AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "YOY(pts)": this.convertZeroValueToBlank(key.MarketShare_VLY),
            CY_M_AB: window.numberWithCommas(key.BookedPassenger_CY),
            LY_M_AB: window.numberWithCommas(key.BookedPassenger_LY),
            VLY_M_AB: window.numberWithCommas(key.BookedPassenger_VLY),
            CY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_CY),
            LY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_LY),
            VLY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_VLY),
            CY_MS_AB: window.numberWithCommas(key.Agent_MS_CY),
            YOY_MS_AB: window.numberWithCommas(key.Agent_MS_YOY),
            AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "YOY(pts)_AB": window.numberWithCommas(key.MarketShare_VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            OD: "Total",
            CY_M: this.convertZeroValueToBlank(key.BookedPassenger_CY),
            LY_M: this.convertZeroValueToBlank(key.BookedPassenger_LY),
            VLY_M: this.convertZeroValueToBlank(key.BookedPassenger_VLY),
            CY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_CY),
            LY_MA: this.convertZeroValueToBlank(key.AirlinesBookedPassenger_LY),
            VLY_MA: this.convertZeroValueToBlank(
              key.AirlinesBookedPassenger_VLY
            ),
            CY_MS: this.convertZeroValueToBlank(key.Agent_MS_CY),
            YOY_MS: this.convertZeroValueToBlank(key.Agent_MS_YOY),
            AL: this.convertZeroValueToBlank(key.MarketShare_CY),
            "YOY(pts)": this.convertZeroValueToBlank(key.MarketShare_VLY),
            CY_M_AB: window.numberWithCommas(key.BookedPassenger_CY),
            LY_M_AB: window.numberWithCommas(key.BookedPassenger_LY),
            VLY_M_AB: window.numberWithCommas(key.BookedPassenger_VLY),
            CY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_CY),
            LY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_LY),
            VLY_MA_AB: window.numberWithCommas(key.AirlinesBookedPassenger_VLY),
            CY_MS_AB: window.numberWithCommas(key.Agent_MS_CY),
            YOY_MS_AB: window.numberWithCommas(key.Agent_MS_YOY),
            AL_AB: window.numberWithCommas(key.MarketShare_CY),
            "YOY(pts)_AB": window.numberWithCommas(key.MarketShare_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return topMarketTable;
  }

  //Distribution Channel Performance
  getChannelPerformance(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue
  ) {
    const url = `${API_URL}/channelperformance?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}`;

    const downloadurl = `${API_URL}/FullYearDownloadChannel?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}`;

    localStorage.setItem("channelDownloadURL", downloadurl);

    var channelData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.response.filter(
          (d) => d.AverageFare_TGT === 0 || d.AverageFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        let passengerzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.response.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.CHANNEL,
                field: "Channel",
                tooltipField: "Channel",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_P",
                tooltipField: "Forecast_P_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_A",
                tooltipField: "Forecast_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_R",
                tooltipField: "Forecast_R_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: string.columnName.MARKET_SHARE,
          //     children: [
          //         { headerName: string.columnName.MARKET_SHARE, field: 'Market Share', tooltipField: "Market Share_AB" },
          //         { headerName: string.columnName.VLY, field: 'VLY_MS', tooltipField: "VLY_MS_AB", cellRenderer: (params) => this.arrowIndicator(params) },
          //         { headerName: string.columnName.SA_SHARE, field: 'SA Share', tooltipField: "SA Share_AB" },
          //         { headerName: string.columnName.VLY, field: 'VLY_SA', tooltipField: "VLY_SA_AB", cellRenderer: (params) => this.arrowIndicator(params) }
          //     ]
          // }
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            Channel: key.ChannelName,
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: this.convertZeroValueToBlank(key.Bookings_TKT),
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            // 'Market Share': this.convertZeroValueToBlank(key.MarketShare_CY),
            // 'VLY_MS': this.convertZeroValueToBlank(key.MarketShare_VLY),
            // 'SA Share': '---',
            // 'VLY_SA': '---',
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
            // 'Market Share_AB': window.numberWithCommas(key.MarketShare_CY),
            // 'VLY_MS_AB': window.numberWithCommas(key.MarketShare_VLY),
            // 'SA Share_AB': '---',
            // 'VLY_SA_AB': '---'
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Channel: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: this.convertZeroValueToBlank(key.Bookings_TKT),
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            // 'Market Share': this.convertZeroValueToBlank(key.MarketShare_CY),
            // 'VLY_MS': this.convertZeroValueToBlank(key.MarketShare_VLY),
            // 'SA Share': '---',
            // 'VLY_SA': '---',
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
            // 'Market Share_AB': window.numberWithCommas(key.MarketShare_CY),
            // 'VLY_MS_AB': window.numberWithCommas(key.MarketShare_VLY),
            // 'SA Share_AB': '---',
            // 'VLY_SA_AB': '---'
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return channelData;
  }

  getSubChannelPerformance(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    channelname,
    getAgent
  ) {
    const url = `${API_URL}/subchannelperformance?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}&channelname=${encodeURIComponent(
      channelname
    )}&getAgent=${encodeURIComponent(getAgent)}`;

    const downloadurl = `${API_URL}/FullYearDownloadsubChannel?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}&channelname=${encodeURIComponent(
      channelname
    )}&getAgent=${encodeURIComponent(getAgent)}`;

    localStorage.setItem("subChannelDownloadURL", downloadurl);

    var subChannelData = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.response.filter(
          (d) => d.AverageFare_TGT === 0 || d.AverageFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        let passengerzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.response.length;
        let firstColumnName = response.data.response[0].columnName;
        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: firstColumnName,
                field: firstColumnName,
                tooltipField: firstColumnName,
                width: 250,
                alignLeft: true,
                underline: firstColumnName !== "CommonOD",
              },
            ],
          },
          {
            headerName: string.columnName.BOOKINGS,
            children: [
              {
                headerName: string.columnName.CY,
                field: "CY_B",
                tooltipField: "CY_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_B",
                tooltipField: "VLY_B_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TKT,
                field: "TKT_B",
                tooltipField: "TKT_B_AB",
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_P",
                tooltipField: "Forecast_P_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_P', tooltipField: "TGT_P_AB",
              //     hide: passengerTGTVisible, sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_P', tooltipField: "VTG_P_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: passengerTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_A",
                tooltipField: "Forecast_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_A', tooltipField: "TGT_A_AB",
              //     hide: avgfareTGTVisible, sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_A', tooltipField: "VTG_A_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: avgfareTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: string.columnName.FORECAST_ACT,
                field: "Forecast_R",
                tooltipField: "Forecast_R_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_R', tooltipField: "TGT_R_AB", hide: revenueTGTVisible,
              //     sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_R', tooltipField: "VTG_R_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: revenueTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          // {
          //     headerName: string.columnName.MARKET_SHARE,
          //     children: [
          //         { headerName: string.columnName.MARKET_SHARE, field: 'Market Share', tooltipField: "Market Share_AB" },
          //         { headerName: string.columnName.VLY, field: 'VLY_MS', tooltipField: "VLY_MS_AB", cellRenderer: (params) => this.arrowIndicator(params) },
          //         { headerName: string.columnName.SA_SHARE, field: 'SA Share', tooltipField: "SA Share_AB" },
          //         { headerName: string.columnName.VLY, field: 'VLY_SA', tooltipField: "VLY_SA_AB", cellRenderer: (params) => this.arrowIndicator(params) }
          //     ]
          // }
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            [firstColumnName]: key[firstColumnName],
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: this.convertZeroValueToBlank(key.Bookings_TKT),
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            // 'Market Share': this.convertZeroValueToBlank(key.MarketShare_CY),
            // 'VLY_MS': this.convertZeroValueToBlank(key.MarketShare_VLY),
            // 'SA Share': '---',
            // 'VLY_SA': '---',
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
            // 'Market Share_AB': window.numberWithCommas(key.MarketShare_CY),
            // 'VLY_MS_AB': window.numberWithCommas(key.MarketShare_VLY),
            // 'SA Share_AB': '---',
            // 'VLY_SA_AB': '---'
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Agents: "Total",
            CY_B: this.convertZeroValueToBlank(key.Bookings_CY),
            VLY_B: this.convertZeroValueToBlank(key.Bookings_VLY),
            TKT_B: this.convertZeroValueToBlank(key.Bookings_TKT),
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            // 'Market Share': this.convertZeroValueToBlank(key.MarketShare_CY),
            // 'VLY_MS': this.convertZeroValueToBlank(key.MarketShare_VLY),
            // 'SA Share': '---',
            // 'VLY_SA': '---',
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
            // 'Market Share_AB': window.numberWithCommas(key.MarketShare_CY),
            // 'VLY_MS_AB': window.numberWithCommas(key.MarketShare_VLY),
            // 'SA Share_AB': '---',
            // 'VLY_SA_AB': '---'
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return subChannelData;
  }

  //Segmentation
  getSegmentationData(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue
  ) {
    const url = `${API_URL}/segmentationreport?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}`;

    const downloadurl = `${API_URL}/FullYearDownloadSegment?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}`;

    localStorage.setItem("segmentationDownloadurl", downloadurl);

    var segmentationreport = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.response.filter(
          (d) => d.AverageFare_TGT === 0 || d.AverageFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        let passengerzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.response.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: "Segment",
                field: "Segment",
                tooltipField: "Segment",
                width: 250,
                alignLeft: true,
                underline: true,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: "Actual",
                field: "Forecast_P",
                tooltipField: "Forecast_P_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_P",
                tooltipField: "TGT_P_AB",
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_P",
                tooltipField: "VTG_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: passengerTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: "Actual",
                field: "Forecast_A",
                tooltipField: "Forecast_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_A",
                tooltipField: "TGT_A_AB",
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_A",
                tooltipField: "VTG_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: avgfareTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: "Actual",
                field: "Forecast_R",
                tooltipField: "Forecast_R_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              {
                headerName: string.columnName.TGT,
                field: "TGT_R",
                tooltipField: "TGT_R_AB",
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VTG,
                field: "VTG_R",
                tooltipField: "VTG_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                hide: revenueTGTVisible,
                sortable: true,
                comparator: this.customSorting,
              },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            Segment: key.Segment,
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            Segment: "Total",
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return segmentationreport;
  }

  getSubSegmentationData(
    endDate,
    startDate,
    regionId,
    countryId,
    cityId,
    getCabinValue,
    segmentValue
  ) {
    const url = `${API_URL}/segmentationreport?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}&segmentValue=${encodeURIComponent(segmentValue)}`;

    const downloadurl = `${API_URL}/FullYearDownloadSegment?endDate=${endDate}&startDate=${startDate}&${FilterParams(
      regionId,
      countryId,
      cityId,
      "Null"
    )}&segmentValue=${encodeURIComponent(segmentValue)}`;

    localStorage.setItem("segmentationReportDownloadURL", downloadurl);

    var subsegmentationreport = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        let avgfarezeroTGT = response.data.response.filter(
          (d) => d.AverageFare_TGT === 0 || d.AverageFare_TGT === null
        );
        let avgfareTGTVisible =
          avgfarezeroTGT.length === response.data.response.length;

        let revenuzeroTGT = response.data.response.filter(
          (d) => d.Revenue_TGT === 0 || d.Revenue_TGT === null
        );
        let revenueTGTVisible =
          revenuzeroTGT.length === response.data.response.length;

        let passengerzeroTGT = response.data.response.filter(
          (d) => d.Passenger_TGT === 0 || d.Passenger_TGT === null
        );
        let passengerTGTVisible =
          passengerzeroTGT.length === response.data.response.length;

        var columnName = [
          {
            headerName: "",
            children: [
              {
                headerName: string.columnName.OD,
                field: "OD",
                tooltipField: "OD",
                width: 250,
                alignLeft: true,
              },
            ],
          },
          {
            headerName: string.columnName.PASSENGER,
            children: [
              {
                headerName: "Actual",
                field: "Forecast_P",
                tooltipField: "Forecast_P_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_P', tooltipField: "TGT_P_AB",
              //     hide: passengerTGTVisible, sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_P', tooltipField: "VTG_P_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: passengerTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_P",
                tooltipField: "VLY_P_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.AVERAGE_FARE_$,
            children: [
              {
                headerName: "Actual",
                field: "Forecast_A",
                tooltipField: "Forecast_A_AB",
                sortable: true,
                comparator: this.customSorting,
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_A', tooltipField: "TGT_A_AB",
              //     hide: avgfareTGTVisible, sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_A', tooltipField: "VTG_A_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: avgfareTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_A",
                tooltipField: "VLY_A_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
          {
            headerName: string.columnName.REVENUE_$,
            headerGroupComponent: "customHeaderGroupComponent",
            children: [
              {
                headerName: "Actual",
                field: "Forecast_R",
                tooltipField: "Forecast_R_AB",
                sortable: true,
                comparator: this.customSorting,
                sort: "desc",
              },
              // {
              //     headerName: string.columnName.TGT, field: 'TGT_R', tooltipField: "TGT_R_AB",
              //     hide: revenueTGTVisible, sortable: true, comparator: this.customSorting
              // },
              // {
              //     headerName: string.columnName.VTG, field: 'VTG_R', tooltipField: "VTG_R_AB",
              //     cellRenderer: (params) => this.arrowIndicator(params), hide: revenueTGTVisible, sortable: true, comparator: this.customSorting
              // },
              {
                headerName: string.columnName.VLY,
                field: "VLY_R",
                tooltipField: "VLY_R_AB",
                cellRenderer: (params) => this.arrowIndicator(params),
                sortable: true,
                comparator: this.customSorting,
              },
            ],
          },
        ];

        var rowData = [];
        response.data.response[0].TableData.forEach((key) => {
          rowData.push({
            OD: key.CommonOD,
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
          });
        });

        var totalData = [];
        response.data.response[0].Total.forEach((key) => {
          totalData.push({
            OD: "Total",
            Forecast_P: this.convertZeroValueToBlank(key.Passenger_CY),
            TGT_P: this.convertZeroValueToBlank(key.Passenger_TGT),
            VTG_P: this.convertZeroValueToBlank(key.Passenger_VTG),
            VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
            Forecast_A: this.convertZeroValueToBlank(key.AverageFare_CY),
            TGT_A: this.convertZeroValueToBlank(key.AverageFare_TGT),
            VTG_A: this.convertZeroValueToBlank(key.AverageFare_VTG),
            VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
            Forecast_R: this.convertZeroValueToBlank(key.Revenue_CY),
            TGT_R: this.convertZeroValueToBlank(key.Revenue_TGT),
            VTG_R: this.convertZeroValueToBlank(key.Revenue_VTG),
            VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
            CY_B_AB: window.numberWithCommas(key.Bookings_CY),
            VLY_B_AB: window.numberWithCommas(key.Bookings_VLY),
            TKT_B_AB: window.numberWithCommas(key.Bookings_TKT),
            Forecast_P_AB: window.numberWithCommas(key.Passenger_CY),
            TGT_P_AB: window.numberWithCommas(key.Passenger_TGT),
            VTG_P_AB: window.numberWithCommas(key.Passenger_VTG),
            VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
            Forecast_A_AB: window.numberWithCommas(key.AverageFare_CY),
            TGT_A_AB: window.numberWithCommas(key.AverageFare_TGT),
            VTG_A_AB: window.numberWithCommas(key.AverageFare_VTG),
            VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
            Forecast_R_AB: window.numberWithCommas(key.Revenue_CY),
            TGT_R_AB: window.numberWithCommas(key.Revenue_TGT),
            VTG_R_AB: window.numberWithCommas(key.Revenue_VTG),
            VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
          });
        });

        return [
          {
            columnName: columnName,
            rowData: rowData,
            totalData: totalData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return subsegmentationreport;
  }

  //RPS POS
  getRPSColumns(section, type, firstColumnName) {
    const userData = Constant.loggedinUser(
      JSON.parse(cookieStorage.getCookie("userDetails"))
    );
    console.log(userData, "uservalidate");
    let cityId = window.localStorage.getItem("CitySelected");
    cityId = cityId && cityId !== "Null" ? true : false;
    const flight = window.localStorage.getItem("FlightSelected");
    const isEdit = (params) => {
      console.log(params, "params");
      if (section === "drilldown" || section === "total") {
        return false;
      } else {
        if (true) {
          if (
            userData.isPOSNetworkAdmin &&
            params.data.drilldown_level === "od"
          ) {
            if (params.data.version_action) {
              console.log("condition1");
              return false;
            } else if (params.data.Approval_level) {
              console.log("condition2");
              return false;
            } else if (userData.accessLevelPOS === params.data.Submit_level) {
              console.log("condition3");
              return false;
            } else if (params.data.Reject_level) {
              console.log("condition4");
              return true;
            } else {
              console.log("condition5");
              return true;
            }
          } else if (
            userData.isPOSNetworkAdmin ||
            userData.isPOSCountryAdmin ||
            (userData.isPOSRegionAdmin &&
              params.data.drilldown_level === "network") ||
            params.data.drilldown_level === "region" ||
            params.data.drilldown_level === "country" ||
            params.data.drilldown_level === "pos"
          ) {
            if (params.data.version_action === "Unfreeze") {
              console.log("conditio6", params.data.version_action);
              return true;
            } else if (params.data.version_action === "Freeze") {
              console.log("conditio10", params.data.version_action);
              // console.log(params.data,'databackend')
              return false;
            } else if (
              userData.accessLevelPOS - 1 ===
              parseInt(params.data.Approval_level)
            ) {
              console.log("condition7");
              return true;
            } else {
              console.log("condition8");
              return false;
            }
          } else {
            console.log("condition9");
            return false;
          }
        }
      }
    };

    const EDIT =
      section === "drilldown" || section === "route" ? "Edited" : "Edit";

    const isSortable = section === "drilldown" ? true : false;
    const customSorting = section === "drilldown" ? this.customSorting : null;

    var columnName = [
      {
        headerName: "",
        children:
          section === "total"
            ? [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 200,
                alignLeft: true,
              },
            ]
            : section === "drilldown"
              ? [
                {
                  headerName: firstColumnName,
                  field: "firstColumnName",
                  tooltipField: "firstColumnName",
                  alignLeft: true,
                  underline:
                    type === "Null" &&
                      firstColumnName !== "O&D" &&
                      firstColumnName !== "Aircraft"
                      ? true
                      : false,
                  width: 200,
                },
              ]
              : [
                {
                  headerName: string.columnName.MONTH,
                  field: "Month",
                  tooltipField: "Comment",
                  width: 200,
                  alignLeft: true,
                  underline: true,
                  checkboxSelection: (params) => {
                    // console.log(userData, params, 'checks')
                    if (userData.isPOSNetworkAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Approval_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSRegionAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Submit_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSCountryAdmin) {
                      if (userData.accessLevelPOS - 1 === 0) {
                        return true;
                      }
                    } else {
                      return false;
                    }
                  },
                },
              ],
      },
      {
        headerName: string.columnName.PASSENGER_OD_RPS,
        children: [
          {
            headerName: "Target",
            field: "FRCT_P",
            tooltipField: "FRCT_P_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: EDIT,
            field: "Edit_P",
            tooltipField: "Edit_P_AB",
            editable: (params) => isEdit(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_P",
            tooltipField: "VLY_P_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.AVERAGE_FARE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_A",
            tooltipField: "FRCT_A_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: EDIT,
            field: "Edit_A",
            tooltipField: "Edit_A_AB",
            editable: (params) => isEdit(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_A",
            tooltipField: "VLY_A_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.REVENUE_$,
        // headerGroupComponent: "customHeaderGroupComponent",
        children: [
          {
            headerName: "Target",
            field: "FRCT_R",
            tooltipField: "FRCT_R_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
            sort: section === "drilldown" ? "desc" : "",
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_R",
            tooltipField: "VLY_R_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
    ];
    return columnName;
  }

  getRPSRouteColumns(section, type, firstColumnName) {
    const userData = Constant.loggedinUser(
      JSON.parse(cookieStorage.getCookie("userDetails"))
    );
    let cityId = window.localStorage.getItem("CitySelected");
    cityId = cityId && cityId !== "Null" ? true : false;
    const flight = window.localStorage.getItem("FlightSelected");
    const isEdit = (params) => {
      if (section === "drilldown") {
        return false;
      } else {
        if (userData.canEditRPS) {
          if (
            userData.isPOSCountryAdmin &&
            params.data.drilldown_level === "pos"
          ) {
            if (params.data.version_action) {
              return false;
            } else if (params.data.Approval_level) {
              return false;
            } else if (userData.accessLevelPOS === params.data.Submit_level) {
              return false;
            } else if (params.data.Reject_level) {
              return true;
            } else {
              return true;
            }
          } else if (
            userData.isPOSNetworkAdmin &&
            params.data.drilldown_level === "network"
          ) {
            if (params.data.version_action === "Unfreeze") {
              return true;
            } else if (
              userData.accessLevelPOS - 1 ===
              parseInt(params.data.Approval_level)
            ) {
              return true;
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
      }
    };

    const EDIT =
      section === "drilldown" || section === "route" ? "Edited" : "Edit";

    const isSortable = section === "drilldown" ? true : false;
    const customSorting = section === "drilldown" ? this.customSorting : null;

    var columnName = [
      {
        headerName: "",
        children:
          section === "total"
            ? [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 200,
                alignLeft: true,
              },
            ]
            : section === "drilldown"
              ? [
                {
                  headerName: firstColumnName,
                  field: "firstColumnName",
                  tooltipField: "firstColumnName",
                  alignLeft: true,
                  underline:
                    type === "Null" &&
                      firstColumnName !== "O&D" &&
                      firstColumnName !== "Aircraft"
                      ? true
                      : false,
                  width: 200,
                },
              ]
              : [
                {
                  headerName: string.columnName.MONTH,
                  field: "Month",
                  tooltipField: "Comment",
                  width: 200,
                  alignLeft: true,
                  underline: true,
                  checkboxSelection: (params) => {
                    // console.log(userData, params, 'checks')
                    if (userData.isPOSNetworkAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Approval_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSRegionAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Submit_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSCountryAdmin) {
                      if (userData.accessLevelPOS - 1 === 0) {
                        return true;
                      }
                    } else {
                      return false;
                    }
                  },
                },
              ],
      },
      {
        headerName: string.columnName.PASSENGER_OD,
        children: [
          {
            headerName: "Target",
            field: "FRCT_P",
            tooltipField: "FRCT_P_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          // { headerName: EDIT, field: 'Edit_P', tooltipField: 'Edit_P_AB', editable: (params) => isEdit(params), sortable: isSortable, comparator: customSorting },
          {
            headerName: string.columnName.VLY,
            field: "VLY_P",
            tooltipField: "VLY_P_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.AVERAGE_FARE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_A",
            tooltipField: "FRCT_A_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          // { headerName: EDIT, field: 'Edit_A', tooltipField: 'Edit_A_AB', editable: (params) => isEdit(params), sortable: isSortable, comparator: customSorting },
          {
            headerName: string.columnName.VLY,
            field: "VLY_A",
            tooltipField: "VLY_A_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.REVENUE_$,
        //headerGroupComponent: 'customHeaderGroupComponent',
        children: [
          {
            headerName: "Target",
            field: "FRCT_R",
            tooltipField: "FRCT_R_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
            sort: section === "drilldown" ? "desc" : "",
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_R",
            tooltipField: "VLY_R_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.RASK_DOLLAR,
        headerTooltip: string.columnName.RASK_DOLLAR,
        children: [
          { headerName: "Target", field: "CY_RK", tooltipField: "CY_RK_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_RK",
            tooltipField: "VLY_RK",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.ASK,
        headerTooltip: string.columnName.ASK,
        children: [
          { headerName: "Target", field: "CY_AK", tooltipField: "CY_AK_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_AK",
            tooltipField: "VLY_AK",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.YIELD,
        headerTooltip: string.columnName.YIELD,
        children: [
          { headerName: "Target", field: "CY_Y", tooltipField: "CY_Y_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_Y",
            tooltipField: "VLY_Y_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.LOAD_FACTOR,
        headerTooltip: string.columnName.LOAD_FACTOR,
        // headerGroupComponent: "customHeaderGroupComponent",
        children: [
          { headerName: "Target", field: "CY_L", tooltipField: "CY_L_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_L",
            tooltipField: "VLY_L_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
    ];
    return columnName;
  }

  getRPSDrilldownColumns(section, type, firstColumnName) {
    const userData = Constant.loggedinUser(
      JSON.parse(cookieStorage.getCookie("userDetails"))
    );
    let cityId = window.localStorage.getItem("CitySelected");
    cityId = cityId && cityId !== "Null" ? true : false;
    const isEdit = (params) => {
      if (section === "drilldown") {
        return false;
      } else {
        if (userData.canEditRPS) {
          if (
            userData.isPOSCountryAdmin &&
            params.data.drilldown_level === "pos"
          ) {
            if (params.data.version_action) {
              return false;
            } else if (params.data.Approval_level) {
              return false;
            } else if (userData.accessLevelPOS === params.data.Submit_level) {
              return false;
            } else if (params.data.Reject_level) {
              return true;
            } else {
              return true;
            }
          } else if (
            userData.isPOSNetworkAdmin &&
            params.data.drilldown_level === "network"
          ) {
            if (params.data.version_action === "Unfreeze") {
              return true;
            } else if (
              userData.accessLevelPOS - 1 ===
              parseInt(params.data.Approval_level)
            ) {
              return true;
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
      }
    };

    const EDIT =
      section === "drilldown" || section === "route" ? "Edited" : "Edit";

    const isSortable = section === "drilldown" ? true : false;
    const customSorting = section === "drilldown" ? this.customSorting : null;

    var columnName = [
      {
        headerName: "",
        children:
          section === "total"
            ? [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 200,
                alignLeft: true,
              },
            ]
            : section === "drilldown"
              ? [
                {
                  headerName: firstColumnName,
                  field: "firstColumnName",
                  tooltipField: "firstColumnName",
                  alignLeft: true,
                  underline:
                    type === "Null" &&
                      firstColumnName !== "O&D" &&
                      firstColumnName !== "Aircraft"
                      ? true
                      : false,
                  width: 200,
                },
              ]
              : [
                {
                  headerName: string.columnName.MONTH,
                  field: "Month",
                  tooltipField: "Comment",
                  width: 200,
                  alignLeft: true,
                  underline: true,
                  checkboxSelection: (params) => {
                    if (userData.isPOSNetworkAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Approval_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSRegionAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Submit_level)
                      ) {
                        return true;
                      }
                    } else {
                      return false;
                    }
                  },
                },
              ],
      },
      {
        headerName: string.columnName.PASSENGER_OD_RPS,
        children: [
          {
            headerName: "Target",
            field: "FRCT_P",
            tooltipField: "FRCT_P_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: EDIT,
            field: "Edit_P",
            tooltipField: "Edit_P_AB",
            editable: (params) => isEdit(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_P",
            tooltipField: "VLY_P_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.AVERAGE_FARE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_A",
            tooltipField: "FRCT_A_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: EDIT,
            field: "Edit_A",
            tooltipField: "Edit_A_AB",
            editable: (params) => isEdit(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_A",
            tooltipField: "VLY_A_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.REVENUE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_R",
            tooltipField: "FRCT_R_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
            sort: section === "drilldown" ? "desc" : "",
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_R",
            tooltipField: "VLY_R_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
    ];
    return columnName;
  }

  getRPSRouteDrilldownColumns(section, type, firstColumnName) {
    const userData = Constant.loggedinUser(
      JSON.parse(cookieStorage.getCookie("userDetails"))
    );
    let cityId = window.localStorage.getItem("CitySelected");
    cityId = cityId && cityId !== "Null" ? true : false;
    const isEdit = (params) => {
      if (section === "drilldown") {
        return false;
      } else {
        if (userData.canEditRPS) {
          if (
            userData.isPOSCountryAdmin &&
            params.data.drilldown_level === "pos"
          ) {
            if (params.data.version_action) {
              return false;
            } else if (params.data.Approval_level) {
              return false;
            } else if (userData.accessLevelPOS === params.data.Submit_level) {
              return false;
            } else if (params.data.Reject_level) {
              return true;
            } else {
              return true;
            }
          } else if (
            userData.isPOSNetworkAdmin &&
            params.data.drilldown_level === "network"
          ) {
            if (params.data.version_action === "Unfreeze") {
              return true;
            } else if (
              userData.accessLevelPOS - 1 ===
              parseInt(params.data.Approval_level)
            ) {
              return true;
            } else {
              return false;
            }
          } else {
            return false;
          }
        }
      }
    };

    const EDIT =
      section === "drilldown" || section === "route" ? "Edited" : "Edit";

    const isSortable = section === "drilldown" ? true : false;
    const customSorting = section === "drilldown" ? this.customSorting : null;

    var columnName = [
      {
        headerName: "",
        children:
          section === "total"
            ? [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                width: 200,
                alignLeft: true,
              },
            ]
            : section === "drilldown"
              ? [
                {
                  headerName: firstColumnName,
                  field: "firstColumnName",
                  tooltipField: "firstColumnName",
                  alignLeft: true,
                  underline:
                    type === "Null" &&
                      firstColumnName !== "O&D" &&
                      firstColumnName !== "Aircraft"
                      ? true
                      : false,
                  width: 200,
                },
              ]
              : [
                {
                  headerName: string.columnName.MONTH,
                  field: "Month",
                  tooltipField: "Comment",
                  width: 200,
                  alignLeft: true,
                  underline: true,
                  checkboxSelection: (params) => {
                    if (userData.isPOSNetworkAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Approval_level)
                      ) {
                        return true;
                      }
                    } else if (userData.isPOSRegionAdmin) {
                      if (
                        userData.accessLevelPOS - 1 ===
                        parseInt(params.data.Submit_level)
                      ) {
                        return true;
                      }
                    } else {
                      return false;
                    }
                  },
                },
              ],
      },
      {
        headerName: string.columnName.PASSENGER_OD,
        children: [
          {
            headerName: "Target",
            field: "FRCT_P",
            tooltipField: "FRCT_P_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          // { headerName: EDIT, field: 'Edit_P', tooltipField: 'Edit_P_AB', editable: (params) => isEdit(params), sortable: isSortable, comparator: customSorting },
          {
            headerName: string.columnName.VLY,
            field: "VLY_P",
            tooltipField: "VLY_P_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.AVERAGE_FARE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_A",
            tooltipField: "FRCT_A_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
          },
          // { headerName: EDIT, field: 'Edit_A', tooltipField: 'Edit_A_AB', editable: (params) => isEdit(params), sortable: isSortable, comparator: customSorting },
          {
            headerName: string.columnName.VLY,
            field: "VLY_A",
            tooltipField: "VLY_A_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.REVENUE_$,
        children: [
          {
            headerName: "Target",
            field: "FRCT_R",
            tooltipField: "FRCT_R_AB",
            width: 250,
            sortable: isSortable,
            comparator: customSorting,
            sort: section === "drilldown" ? "desc" : "",
          },
          {
            headerName: string.columnName.VLY,
            field: "VLY_R",
            tooltipField: "VLY_R_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: string.columnName.RASK_DOLLAR,
        headerTooltip: string.columnName.RASK_DOLLAR,
        children: [
          { headerName: "Target", field: "CY_RK", tooltipField: "CY_RK_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_RK",
            tooltipField: "VLY_RK_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.ASK,
        headerTooltip: string.columnName.ASK,
        children: [
          { headerName: "Target", field: "CY_AK", tooltipField: "CY_AK_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_AK",
            tooltipField: "VLY_AK_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.YIELD,
        headerTooltip: string.columnName.YIELD,
        children: [
          { headerName: "Target", field: "CY_Y", tooltipField: "CY_Y_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_Y",
            tooltipField: "VLY_Y_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
      {
        headerName: string.columnName.LOAD_FACTOR,
        headerTooltip: string.columnName.LOAD_FACTOR,
        // headerGroupComponent: 'customHeaderGroupComponent',
        children: [
          { headerName: "Target", field: "CY_L", tooltipField: "CY_L_AB" },
          {
            headerName: string.columnName.VLY,
            field: "VLY_L",
            tooltipField: "VLY_L_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            width: 250,
          },
        ],
      },
    ];
    return columnName;
  }

  getRPSData(data, section, drilldown_level) {
    const rowData = [];
    data.forEach((key, i) => {
      rowData.push({
        Month: section === "total" ? "Total" : key.MonthName,
        firstColumnName: section === "total" ? "Total" : key.ColumnName,
        // "CY_C": this.convertZeroValueToBlank(34455533),
        // "LY_C": this.convertZeroValueToBlank(34455533),
        FRCT_P: this.convertZeroValueToBlank(key.Passenger_CY),
        Edit_P: window.numberWithCommas(key.Passenger_EDIT, 2),
        VLY_P: this.convertZeroValueToBlank(key.Passenger_VLY),
        FRCT_A: this.convertZeroValueToBlank(key.AverageFare_CY),
        Edit_A: window.numberWithCommas(key.AverageFare_EDIT, 2),
        VLY_A: this.convertZeroValueToBlank(key.AverageFare_VLY),
        FRCT_R: this.convertZeroValueToBlank(key.Revenue_CY),
        VLY_R: this.convertZeroValueToBlank(key.Revenue_VLY),
        CY_AK: window.numberWithCommas(key.ASK_TGT),
        VLY_AK: this.convertZeroValueToBlank(key.ASK_VLY),
        CY_RK: this.convertZeroValueToBlank(key.RASK_TGT),
        VLY_RK: this.convertZeroValueToBlank(key.RASK_VLY),
        CY_Y: window.numberWithCommas(key.yield_TGT),
        VLY_Y: this.convertZeroValueToBlank(key.yield_VLY),
        CY_L: this.convertZeroValueToBlank(key.LoadFactor_TGT),
        VLY_L: this.convertZeroValueToBlank(key.LoadFactor_VLY),

        // 'LY_AL': this.convertZeroValueToBlank(key.Revenue_VLY),
        // "CY_C_AB": window.numberWithCommas(34455533),
        // "LY_C_AB": window.numberWithCommas(34455533),

        FRCT_P_AB: window.numberWithCommas(key.Passenger_CY),
        // 'Edit_P_AB': window.numberWithCommas(key.Passenger_EDIT),
        VLY_P_AB: window.numberWithCommas(key.Passenger_VLY),
        FRCT_A_AB: window.numberWithCommas(key.AverageFare_CY),
        // 'Edit_A_AB': window.numberWithCommas(key.AverageFare_EDIT),
        VLY_A_AB: window.numberWithCommas(key.AverageFare_VLY),
        FRCT_R_AB: window.numberWithCommas(key.Revenue_CY),
        VLY_R_AB: window.numberWithCommas(key.Revenue_VLY),
        CY_AK_AB: window.numberWithCommas(key.ASK_TGT),
        VLY_AK_AB: this.convertZeroValueToBlank(key.ASK_VLY),
        CY_RK_AB: this.convertZeroValueToBlank(key.RASK_TGT),
        VLY_RK_AB: this.convertZeroValueToBlank(key.RASK_VLY),
        CY_Y_AB: window.numberWithCommas(key.yield_TGT),
        VLY_Y_AB: this.convertZeroValueToBlank(key.yield_VLY),
        CY_L_AB: this.convertZeroValueToBlank(key.LoadFactor_TGT),
        VLY_L_AB: this.convertZeroValueToBlank(key.LoadFactor_VLY),
        // 'LY_AL_AB': window.numberWithCommas(key.Revenue_VLY),
        Year: key.Year,
        MonthName: key.Monthfullname,
        Action: key.Action,
        Submit_level: key.Submit_level
          ? parseInt(key.Submit_level)
          : key.Submit_level,
        Approval_level: key.Approval_level
          ? parseInt(key.Approval_level)
          : key.Approval_level,
        Reject_level: key.Reject_level
          ? parseInt(key.Reject_level)
          : key.Reject_level,
        drilldown_level: drilldown_level,
        version_action: key.version_action,
      });
    });
    return rowData;
  }

  getRPSMonthTables(
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    selectedVersion
  ) {
    console.log("firstcheck");
    const url = `${API_URL}/rpsmonthly?${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(commonOD)}&version=${selectedVersion}`;
    var result = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const userData = Constant.loggedinUser(
          JSON.parse(cookieStorage.getCookie("userDetails"))
        );
        const responseData = response.data.response;
        let isActionPerformed = false;
        let isAllApproved = false;
        let count = 0;
        responseData.TableData.forEach((d) => {
          if (!isActionPerformed) {
            if (userData.isPOSNetworkAdmin) {
              if (userData.accessLevelPOS - 1 === parseInt(d.Approval_level)) {
                isActionPerformed = true;
              }
            } else if (userData.isPOSRegionAdmin) {
              if (userData.accessLevelPOS - 1 === parseInt(d.Submit_level)) {
                isActionPerformed = true;
              }
            }
          }
          if (parseInt(d.Approval_level) === 3) {
            count = count + 1;
          }
        });
        isAllApproved = count === responseData.TableData.length ? true : false;

        return [
          {
            columnName: this.getRPSColumns(),
            rowData: this.getRPSData(
              responseData.TableData,
              "",
              Constant.DrillDownLevel(regionId, countryId, cityId)
            ),
            totalData: this.getRPSData(responseData.Total, "total"),
            columnNameTotal: this.getRPSColumns("total"),
            apiMonthlyData: responseData,
            isActionPerformed: isActionPerformed,
            isAllApproved: isAllApproved,
            // "currentAccess": response.data.CurretAccess
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return result;
  }

  getRPSDrillDown(
    getYear,
    gettingMonth,
    regionId,
    countryId,
    cityId,
    commonOD,
    getCabinValue,
    type,
    selectedVersion,
    odsearchvalue
  ) {
    const url = `${API_URL}/rpsdrilldown?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(
      commonOD
    )}&type=${type}&version=${selectedVersion}&odSearch=${odsearchvalue}`;
    console.log("Year:::::::", getYear);

    const downloadurl = `${API_URL}/FullYearRPS_POS?getYear=${getYear}&gettingMonth=${gettingMonth}&${Params(
      regionId,
      countryId,
      cityId,
      getCabinValue
    )}&commonOD=${encodeURIComponent(
      commonOD
    )}&type=${type}&version=${selectedVersion}`;

    localStorage.setItem("RPSPOSDownloadURL", downloadurl);

    var result = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data.response;
        const firstColumnName = responseData.ColumName;
        return [
          {
            columnName: this.getRPSDrilldownColumns(
              "drilldown",
              type,
              firstColumnName
            ),
            rowData: this.getRPSData(responseData.TableData),
            totalData: this.getRPSData(responseData.Total, "total"),
            currentAccess: responseData.CurrentAccess,
            tabName: responseData.ColumName,
            firstTabName: responseData.first_ColumName,
            apiDrilldownData: responseData,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return result;
  }

  //RPS Route
  getRPSRouteMonthTables(
    currency,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue,
    version
  ) {
    const url = `${API_URL}/rpsmonthlyroute?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&version=${version}`;
    var routemonthtable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data.response;
        return [
          {
            columnName: this.getRPSRouteColumns("route"),
            rowData: this.getRPSData(responseData.TableData),
            totalData: this.getRPSData(responseData.Total, "total"),
            currentAccess: response.data.CurretAccess,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log("error", error);
      });

    return routemonthtable;
  }

  getRPSRouteDrillDownData(
    getYear,
    currency,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue,
    type,
    version
  ) {
    const url = `${API_URL}/rpsdrilldownroute?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&type=${type}&version=${version}`;

    const downloadurl = `${API_URL}/FullYearRPS_Route?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&selectedLeg=${encodeURIComponent(leg)}&flight=${String.removeQuotes(
      flight
    )}&type=${type}&version=${version}`;

    localStorage.setItem("RPSRouteDownloadURL", downloadurl);

    var routeRegionTable = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data.response;
        const firstColumnName = responseData.ColumName;
        return [
          {
            columnName: this.getRPSRouteDrilldownColumns(
              "drilldown",
              type,
              firstColumnName
            ),
            rowData: this.getRPSData(responseData.TableData),
            totalData: this.getRPSData(responseData.Total, "total"),
            currentAccess: responseData.CurrentAccess,
            tabName: responseData.ColumName,
            firstTabName: responseData.first_ColumName,
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return routeRegionTable;
  }

  //Route Profitability Solutions
  getRPSolutionColumn(section, type, firstColumnName) {
    const isSortable = section === "drilldown" ? true : false;
    const customSorting = section === "drilldown" ? this.customSorting : null;
    var columnName = [
      {
        headerName: "",
        children:
          section === "drilldown"
            ? [
              {
                headerName: firstColumnName,
                field: "firstColumnName",
                tooltipField: "firstColumnName",
                alignLeft: true,
                underline:
                  type === "Null" && firstColumnName !== "Aircraft"
                    ? true
                    : false,
              },
            ]
            : [
              {
                headerName: string.columnName.MONTH,
                field: "Month",
                tooltipField: "Month",
                alignLeft: true,
                underline: true,
              },
            ],
      },
      {
        headerName: "VC contri(RM'Mil)",
        headerTooltip: "VC contri(RM'Mil)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_VC_Contri",
            tooltipField: "#_VC_Contri_AB",
            underline: section === "drilldown" ? false : true,
            sortable: isSortable,
            comparator: customSorting,
            sort: section === "drilldown" ? "desc" : "",
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_VC_Contri",
            tooltipField: "Bgt_VC_Contri_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_VC_Contri",
            tooltipField: "LY_VC_Contri_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "DOC contri(RM'Mil)",
        headerTooltip: "DOC contri(RM'Mil)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_DOC_Contri",
            tooltipField: "#_DOC_Contri_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_DOC_Contri",
            tooltipField: "Bgt_DOC_Contri_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_DOC_Contri",
            tooltipField: "LY_DOC_Contri_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Total Cost(RM'Mil)",
        headerTooltip: "Total Cost(RM'Mil)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_TC",
            tooltipField: "#_TC_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_TC",
            tooltipField: "Bgt_TC_AB",
            cellRenderer: (params) => this.costArrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_TC",
            tooltipField: "LY_TC_AB",
            cellRenderer: (params) => this.costArrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Total Revenue(RM'Mil)",
        headerTooltip: "Total Revenue(RM'Mil)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_TR",
            tooltipField: "#_TR_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_TR",
            tooltipField: "Bgt_TR_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_TR",
            tooltipField: "LY_TR_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Surplus/Deficit(RM'Mil)",
        headerTooltip: "Surplus/Deficit(RM'Mil)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "CY_SD",
            tooltipField: "CY_SD_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_SD",
            tooltipField: "Bgt_SD_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_SD",
            tooltipField: "LY_SD_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Total RASK(RM'sen)",
        headerTooltip: "Total RASK(RM'sen)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_RASK",
            tooltipField: "#_RASK_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_RASK",
            tooltipField: "Bgt_RASK_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_RASK",
            tooltipField: "LY_RASK_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Total CASK(RM'sen)",
        headerTooltip: "Total CASK(RM'sen)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_CASK",
            tooltipField: "#_CASK_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_CASK",
            tooltipField: "Bgt_CASK_AB",
            cellRenderer: (params) => this.costArrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_CASK",
            tooltipField: "LY_CASK_AB",
            cellRenderer: (params) => this.costArrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },

      {
        headerName: "Total Yield(RM'sen)",
        headerTooltip: "Total Yield(RM'sen)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_Pax",
            tooltipField: "#_Pax_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_Pax",
            tooltipField: "Bgt_Pax_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_Pax",
            tooltipField: "LY_Pax_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
      {
        headerName: "Load Factor(Overall)(%)",
        headerTooltip: "Load Factor(Overall)(%)",
        children: [
          {
            headerName: string.columnName.CY,
            field: "#_BL",
            tooltipField: "#_BL_AB",
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs Bgt",
            field: "Bgt_BL",
            tooltipField: "Bgt_BL_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
          {
            headerName: "vs LY",
            field: "LY_BL",
            tooltipField: "LY_BL_AB",
            cellRenderer: (params) => this.arrowIndicator(params),
            sortable: isSortable,
            comparator: customSorting,
          },
        ],
      },
    ];
    return columnName;
  }

  getRPSolutionData(data, section) {
    var rowData = [];

    data.forEach((key) => {
      rowData.push({
        Month:
          section === "total"
            ? "Total"
            : section === "dayLevel"
              ? key.Day
              : key.MonthName,
        firstColumnName: section === "total" ? "Total" : key.ColumnName,
        Day: section,
        "#_VC_Contri": this.convertZeroValueToBlank(key.VC_CY),
        Bgt_VC_Contri: this.convertZeroValueToBlank(key.VC_VTG),
        LY_VC_Contri: this.convertZeroValueToBlank(key.VC_VLY),
        "#_DOC_Contri": this.convertZeroValueToBlank(key.DOC_CY),
        Bgt_DOC_Contri: this.convertZeroValueToBlank(key.DOC_VTG),
        LY_DOC_Contri: this.convertZeroValueToBlank(key.DOC_VLY),
        "#_TC": this.convertZeroValueToBlank(key.Total_Cost_CY),
        Bgt_TC: this.convertZeroValueToBlank(key.Total_Cost_VTG),
        LY_TC: this.convertZeroValueToBlank(key.Total_Cost_VLY),
        "#_RASK": this.convertZeroValueToBlank(key.Rask_CY),
        Bgt_RASK: this.convertZeroValueToBlank(key.Rask_VTG),
        LY_RASK: this.convertZeroValueToBlank(key.Rask_VLY),
        "#_CASK": this.convertZeroValueToBlank(key.CASK_CY),
        Bgt_CASK: this.convertZeroValueToBlank(key.CASK_VTG),
        LY_CASK: this.convertZeroValueToBlank(key.CASK_VLY),
        "#_Pax": this.convertZeroValueToBlank(key.Pax_Yield_CY),
        Bgt_Pax: this.convertZeroValueToBlank(key.Pax_Yield_VTG),
        LY_Pax: this.convertZeroValueToBlank(key.Pax_Yield_VLY),
        CY_SD: this.convertZeroValueToBlank(key.Surplus_Deficit_CY),
        Bgt_SD: this.convertZeroValueToBlank(key.Surplus_Deficit_VTG),
        LY_SD: this.convertZeroValueToBlank(key.Surplus_Deficit_VLY),
        "#_TR": this.convertZeroValueToBlank(key.Total_Revenue_CY),
        Bgt_TR: this.convertZeroValueToBlank(key.Total_Revenue_VTG),
        LY_TR: this.convertZeroValueToBlank(key.Total_Revenue_VLY),
        "#_BL": this.convertZeroValueToBlank(key.Breakeven_Loadfactor_CY),
        Bgt_BL: this.convertZeroValueToBlank(key.Breakeven_Loadfactor_VTG),
        LY_BL: this.convertZeroValueToBlank(key.Breakeven_Loadfactor_VLY),
        "#_VC_Contri_AB": window.numberWithCommas(key.VC_CY),
        Bgt_VC_Contri_AB: window.numberWithCommas(key.VC_VTG),
        LY_VC_Contri_AB: window.numberWithCommas(key.VC_VLY),
        "#_DOC_Contri_AB": window.numberWithCommas(key.DOC_CY),
        Bgt_DOC_Contri_AB: window.numberWithCommas(key.DOC_VTG),
        LY_DOC_Contri_AB: window.numberWithCommas(key.DOC_VLY),
        "#_TC_AB": window.numberWithCommas(key.Total_Cost_CY),
        Bgt_TC_AB: window.numberWithCommas(key.Total_Cost_VTG),
        LY_TC_AB: window.numberWithCommas(key.Total_Cost_VLY),
        "#_RASK_AB": window.numberWithCommas(key.Rask_CY),
        Bgt_RASK_AB: window.numberWithCommas(key.Rask_VTG),
        LY_RASK_AB: window.numberWithCommas(key.Rask_VLY),
        "#_CASK_AB": window.numberWithCommas(key.CASK_CY),
        Bgt_CASK_AB: window.numberWithCommas(key.CASK_VTG),
        LY_CASK_AB: window.numberWithCommas(key.CASK_VLY),
        "#_Pax_AB": window.numberWithCommas(key.Pax_Yield_CY),
        Bgt_Pax_AB: window.numberWithCommas(key.Pax_Yield_VTG),
        LY_Pax_AB: window.numberWithCommas(key.Pax_Yield_VLY),
        CY_SD_AB: window.numberWithCommas(key.Surplus_Deficit_CY),
        Bgt_SD_AB: window.numberWithCommas(key.Surplus_Deficit_VTG),
        LY_SD_AB: window.numberWithCommas(key.Surplus_Deficit_VLY),
        "#_TR_AB": window.numberWithCommas(key.Total_Revenue_CY),
        Bgt_TR_AB: window.numberWithCommas(key.Total_Revenue_VTG),
        LY_TR_AB: window.numberWithCommas(key.Total_Revenue_VLY),
        "#_BL_AB": window.numberWithCommas(key.Breakeven_Loadfactor_CY),
        Bgt_BL_AB: window.numberWithCommas(key.Breakeven_Loadfactor_VTG),
        LY_BL_AB: window.numberWithCommas(key.Breakeven_Loadfactor_VLY),
        Year: key.Year,
        MonthName: key.monthfullname,
      });
    });

    return rowData;
  }

  getRouteProfitabilitySolutionMonthlyData(
    currency,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/routeprofitabilitymonthly?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;

    const downloadurl = `${API_URL}/routeprofitabilitymonthly?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;

    localStorage.setItem("RouteProfitabilityMonthlyDownloadURL", downloadurl);

    var result = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data.response;
        return [
          {
            columnName: this.getRPSolutionColumn(),
            rowData: this.getRPSolutionData(responseData.TableData),
            currentAccess: responseData.CurrentAccess,
            totalData: this.getRPSolutionData(responseData.Total, "total"),
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log("error", error);
      });

    return result;
  }

  getRouteProfitabilitySolutionDayData(
    currency,
    getYear,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/rrmonthlydropDown?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;
    var result = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data;

        return [
          {
            rowData: this.getRPSolutionData(responseData.TableData, "dayLevel"),
          },
        ]; // the response.data is string of src
      })
      .catch((error) => {
        console.log("error", error);
      });

    return result;
  }

  getRouteProfitabilitySolutionDrillDownData(
    getYear,
    currency,
    gettingMonth,
    gettingDay,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue,
    type
  ) {
    const url = `${API_URL}/routeprofitabilitydrilldown?getYear=${getYear}&gettingMonth=${gettingMonth}&getDay=${gettingDay}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}&type=${type}`;

    const downloadurl = `${API_URL}/routeprofitabilitydrilldown?getYear=${getYear}&gettingMonth=${gettingMonth}&getDay=${gettingDay}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}&type=${type}`;

    localStorage.setItem("RouteProfitabilityDrillDownDownloadURL", downloadurl);

    var result = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const responseData = response.data.response;
        const firstColumnName = responseData.ColumnName;

        return [
          {
            columnName: this.getRPSolutionColumn(
              "drilldown",
              type,
              firstColumnName
            ),
            rowData: this.getRPSolutionData(responseData.TableData),
            currentAccess: responseData.CurrentAccess,
            tabName: responseData.ColumnName,
            firstTabName: responseData.first_ColumnName,
            totalData: this.getRPSolutionData(responseData.Total, "total"),
          },
        ]; // the responseData is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return result;
  }

  exportCSVRouteProfitabilitySolutionMonthlyURL(
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue
  ) {
    const url = `routeprofitabilitymonthly?selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;
    return url;
  }

  exportCSVRouteProfitabilitySolutionDrillDownURL(
    getYear,
    gettingMonth,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue,
    type
  ) {
    const url = `routeprofitabilitydrilldown?getYear=${getYear}&gettingMonth=${gettingMonth}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}&type=${type}`;
    return url;
  }

  //Route Profitability Consolidate
  getRPCColumns() {
    const arrowIndicator = (params) => {
      if (params.data.cost === "cost") {
        return this.costArrowIndicator(params);
      } else if (params.data.cost === "commercial_cost") {
        if (params.data.Name.toLowerCase().includes("cask")) {
          return this.costArrowIndicator(params);
        } else {
          return this.arrowIndicator(params);
        }
      } else {
        return this.arrowIndicator(params);
      }
    };

    var columnName = [
      {
        headerName: "",
        children: [
          {
            headerName: " ",
            field: "Name",
            tooltipField: "Name",
            alignLeft: true,
          },
        ],
      },
      {
        headerName: "Daily",
        headerTooltip: "Daily",
        children: [
          {
            headerName: string.columnName.ACTUAL,
            field: "ACTUAL_D",
            tooltipField: "ACTUAL_D_AB",
          },
          {
            headerName: "Budget",
            field: "Budget_D",
            tooltipField: "Budget_D_AB",
          },
          {
            headerName: "Last Year",
            field: "Last_Year_D",
            tooltipField: "Last_Year_D_AB",
          },
          {
            headerName: "VBGT",
            field: "VAR_BUDGET_D",
            tooltipField: "VAR_BUDGET_D_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
          {
            headerName: "VLY",
            field: "VAR_LAST_D",
            tooltipField: "VAR_LAST_D_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
        ],
      },
      {
        headerName: "Month to Date",
        headerTooltip: "Month to Date",
        children: [
          {
            headerName: string.columnName.ACTUAL,
            field: "ACTUAL_M",
            tooltipField: "ACTUAL_M_AB",
          },
          {
            headerName: "Budget",
            field: "Budget_M",
            tooltipField: "Budget_M_AB",
          },
          {
            headerName: "Last Year",
            field: "Last_Year_M",
            tooltipField: "Last_Year_M_AB",
          },
          {
            headerName: "VBGT",
            field: "VAR_BUDGET_M",
            tooltipField: "VAR_BUDGET_M_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
          {
            headerName: "VLY",
            field: "VAR_LAST_M",
            tooltipField: "VAR_LAST_M_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
        ],
      },
      {
        headerName: "Year To Date",
        headerTooltip: "Year To Date",
        children: [
          {
            headerName: string.columnName.ACTUAL,
            field: "ACTUAL_Y",
            tooltipField: "ACTUAL_Y_AB",
          },
          {
            headerName: "Budget",
            field: "Budget_Y",
            tooltipField: "Budget_Y_AB",
          },
          {
            headerName: "Last Year",
            field: "Last_Year_Y",
            tooltipField: "Last_Year_Y_AB",
          },
          {
            headerName: "VBGT",
            field: "VAR_BUDGET_Y",
            tooltipField: "VAR_BUDGET_Y_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
          {
            headerName: "VLY",
            field: "VAR_LAST_Y",
            tooltipField: "VAR_LAST_Y_AB",
            cellRenderer: (params) => arrowIndicator(params),
          },
        ],
      },
    ];

    return columnName;
  }

  getRPCData(responseData, total, isCost) {
    const rowData = [];
    responseData.map((key) => {
      rowData.push({
        Name: total ? `Total ${total}` : key.Name,
        ACTUAL_D: this.convertZeroValueToBlank(key.CY_Daily),
        Budget_D: this.convertZeroValueToBlank(key.TG_Daily),
        Last_Year_D: this.convertZeroValueToBlank(key.LY_Daily),
        VAR_BUDGET_D: this.convertZeroValueToBlank(key.VTG_Daily),
        VAR_LAST_D: this.convertZeroValueToBlank(key.VLY_Daily),
        ACTUAL_M: this.convertZeroValueToBlank(key.CY_MTD),
        Budget_M: this.convertZeroValueToBlank(key.TG_MTD),
        Last_Year_M: this.convertZeroValueToBlank(key.LY_MTD),
        VAR_BUDGET_M: this.convertZeroValueToBlank(key.VTG_MTD),
        VAR_LAST_M: this.convertZeroValueToBlank(key.VLY_MTD),
        ACTUAL_Y: this.convertZeroValueToBlank(key.CY_YTD),
        Budget_Y: this.convertZeroValueToBlank(key.TG_YTD),
        Last_Year_Y: this.convertZeroValueToBlank(key.LY_YTD),
        VAR_BUDGET_Y: this.convertZeroValueToBlank(key.VTG_YTD),
        VAR_LAST_Y: this.convertZeroValueToBlank(key.VLY_YTD),
        ACTUAL_D_AB: window.numberWithCommas(key.CY_Daily),
        Budget_D_AB: window.numberWithCommas(key.TG_Daily),
        Last_Year_D_AB: window.numberWithCommas(key.LY_Daily),
        VAR_BUDGET_D_AB: window.numberWithCommas(key.VTG_Daily),
        VAR_LAST_D_AB: window.numberWithCommas(key.VLY_Daily),
        ACTUAL_M_AB: window.numberWithCommas(key.CY_MTD),
        Budget_M_AB: window.numberWithCommas(key.TG_MTD),
        Last_Year_M_AB: window.numberWithCommas(key.LY_MTD),
        VAR_BUDGET_M_AB: window.numberWithCommas(key.VTG_MTD),
        VAR_LAST_M_AB: window.numberWithCommas(key.VLY_MTD),
        ACTUAL_Y_AB: window.numberWithCommas(key.CY_YTD),
        Budget_Y_AB: window.numberWithCommas(key.TG_YTD),
        Last_Year_Y_AB: window.numberWithCommas(key.LY_YTD),
        VAR_BUDGET_Y_AB: window.numberWithCommas(key.VTG_YTD),
        VAR_LAST_Y_AB: window.numberWithCommas(key.VLY_YTD),
        cost: isCost ? isCost : null,
      });
    });

    return rowData;
  }

  getRPCHeader() {
    const months = [
      "Passenger",
      "Hard Frieght",
      "Excess Baggage",
      "Express",
      "Mail",
      "Others",
    ];
    const rowData = [];
    months.map((key) => {
      rowData.push({
        Name: key,
      });
    });
    return [
      {
        columnName: this.getRPCColumns(),
        rowData: rowData,
        // "currentAccess": response.data.CurretAccess
      },
    ]; // the response.data is string of src
  }

  getRPNonCommercialInfo(
    getYear,
    gettingMonth,
    gettingDay,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/routeprofitabilitynoncommercialtable?getYear=${getYear}&gettingMonth=${gettingMonth}&getDay=${gettingDay}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;

    localStorage.setItem("RouteProfitabilityNCDownloadURL", url);

    var rpNonCommercial = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const rows = response.data.response;
        const TableData = rows.TableData;

        return [
          {
            vcContribution: this.getRPCData(rows.VC_Contribution),
            docContribution: this.getRPCData(rows.DOC_Contribution),
            revenue: this.getRPCData(TableData.Revenue),
            revenueTotal: this.getRPCData(rows.Total_Revenue, "Revenue"),
            vc: this.getRPCData(TableData.VC, "", "cost"),
            vcTotal: this.getRPCData(rows.Total_VC, "Variable Cost", "cost"),
            doc: this.getRPCData(TableData.DOC, "", "cost"),
            docTotal: this.getRPCData(
              rows.Total_DOC,
              "Direct Fixed Cost",
              "cost"
            ),
            totalOtherFixCost: this.getRPCData(
              rows.Total_Other_Fixed_Cost,
              "",
              "cost"
            ),
            totalCost: this.getRPCData(rows.Total_Cost, "", "cost"),
            surplusDeficit: this.getRPCData(rows.Surplus_Deficit),
            noneRoute: this.getRPCData(rows.None_Route),
            NIAT: this.getRPCData(rows.NIAT),
          },
        ]; // the responseData is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return rpNonCommercial;
  }

  getRPCommercialInfo(
    getYear,
    gettingMonth,
    gettingDay,
    routeGroup,
    regionId,
    countryId,
    routeId,
    flight,
    getCabinValue
  ) {
    const url = `${API_URL}/routeprofitabilitycommercialtable?getYear=${getYear}&gettingMonth=${gettingMonth}&getDay=${gettingDay}&selectedRouteGroup=${routeGroup}&${ROUTEParams(
      regionId,
      countryId,
      routeId,
      getCabinValue
    )}&flight=${String.removeQuotes(flight)}`;

    localStorage.setItem("RouteProfitabilityCDownloadURL", url);

    var rpCommercial = axios
      .get(url, this.getDefaultHeader())
      .then((response) => {
        const TableData = response.data.response.TableData;
        const Total = [response.data.response.Total];

        return [
          {
            rowData: this.getRPCData(TableData, "", "commercial_cost"),
            total: this.getRPCData(Total, "Commercial Info"),
          },
        ]; // the responseData is string of src
      })
      .catch((error) => {
        this.errorHandling(error);
      });

    return rpCommercial;
  }
}
