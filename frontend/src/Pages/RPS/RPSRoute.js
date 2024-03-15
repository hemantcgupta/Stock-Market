import React, { Component } from "react";
import APIServices from "../../API/apiservices";
import eventApi from "../../API/eventApi";
import ChartModelDetails from "../../Component/chartModel";
import DatatableModelDetails from "../../Component/dataTableModel";
import DataTableComponent from "../../Component/DataTableComponent";
import DownloadCSV from "../../Component/DownloadCSV";
import Loader from "../../Component/Loader";
import TotalRow from "../../Component/TotalRow";
import color from "../../Constants/color";
import $ from "jquery";
import "../../App.scss";
import "./RPS.scss";
import TopMenuBar from "../../Component/TopMenuBar";
import cookieStorage from "../../Constants/cookie-storage";
import Constant from "../../Constants/validator";
import Input from "@material-ui/core/Input";
import Pagination from "../../Component/pagination";
import RPSRouteDownloadCustomHeaderGroup from "./RPSRouteDownloadCustomHeaderGroup copy";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import ListItemText from "@material-ui/core/ListItemText";
import Select from "@material-ui/core/Select";
import Checkbox from "@material-ui/core/Checkbox";
import BrowserToProps from "react-browser-to-props";
import Modal from "react-bootstrap-modal";

const apiServices = new APIServices();

const currentYear = new Date().getFullYear();
let monthData = [];

let bcData = [];
let baseAccess = "";

class RPSRoute extends Component {
  constructor(props) {
    super(props);
    this.pathName = window.location.pathname;
    this.userDetails = JSON.parse(cookieStorage.getCookie("userDetails"));
    this.selectedRegion = null;
    this.selectedCountry = null;
    this.selectedRoute = null;
    this.gridApiMonth = null;
    this.state = {
      showLastYearRows: false,
      showNextYearRows: false,
      routeMonthRowClassRule: {
        "highlight-row": "data.highlightMe",
      },
      monthRowData: [],
      monthColumn: [],
      monthTotalData: [],
      drillDownTotalData: [],
      drillDownData: [],
      drillDownColumn: [],
      segmentData: [],
      segmentColumn: [],
      odData: [],
      odColumn: [],
      modelRegionDatas: [],
      modelregioncolumn: [],
      modalData: [],
      modalCompartmentColumn: [],
      modalCompartmentData: [],
      tableDatas: true,
      gettingMonth: null,
      gettingYear: null,
      monthTableTitle: "",
      tableTitle: "",
      tabLevel: "",
      cabinOption: [],
      getCabinValue: [],
      cabinSelectedDropDown: [],
      cabinDisable: true,
      currency: "bc",
      chartVisible: false,
      tableModalVisible: false,
      tabName: "Region",
      regionId: "*",
      countryId: "*",
      routeId: "*",
      leg: "*",
      flight: "*",
      type: "Null",
      baseAccess: "",
      routeGroup: "",
      accessLevelDisable: false,
      selectedData: "Null",
      loading: false,
      loading2: false,
      loading3: false,
      firstLoadList: false,
      firstHome: true,
      posContributionModal: false,
      outerTab: "",
      ancillaryParam: false,
      routeParam: false,
      ensureIndexVisible: null,
      posContributionTable: "OD",
      posContributionTableHeader: "",
      currentPage: "",
      totalPages: "",
      totalRecords: "",
      paginationStart: 1,
      paginationEnd: "",
      paginationSize: "",
      count: 1,
      loading: false,
      regionLevelAccess: false,
      version: "Null",
      Rpsversion: "Null",
    };
    this.sendEvent("1", "User viewed Route Page", "/route", "Route Page");
  }

  sendEvent = (id, description, path, page) => {
    var eventData = {
      event_id: id,
      description: description,
      where_path: path,
      page_name: page,
    };
    eventApi.sendEvent(eventData);
  };

  componentDidMount() {
    var self = this;
    const ancillary = Constant.getParameterByName(
      "ancillaryRoute",
      window.location.href
    );
    const route = Constant.getParameterByName("route", window.location.href);
    this.setState({
      ancillaryParam: ancillary ? ancillary : false,
      routeParam: route ? route : false,
    });
    self.getFiltersValue();

    apiServices.getClassNameDetails().then((result) => {
      if (result) {
        var classData = result[0].classDatas;
        self.setState({ cabinOption: classData, cabinDisable: false });
      }
    });
  }

  componentDidUpdate() {
    window.onpopstate = (e) => {
      const obj = this.props.browserToProps.queryParams;
      let data = Object.values(obj);
      let title = Object.keys(obj);
      const lastIndex = title.length - 1;
      if (data[0] !== "undefined") {
        this.pushURLToBcData(obj, title, data, lastIndex);
        this.setState({ firstHome: true });
      } else {
        if (this.state.firstHome) {
          this.homeHandleClick();
        }
      }
    };
  }

  pushURLToBcData(obj, title, data, lastIndex) {
    const self = this;
    let group = [];
    let region = [];
    let country = [];
    let city = [];

    let routeGroup = obj["RouteGroup"];
    this.setState({ routeGroup: routeGroup });
    window.localStorage.setItem(
      "RouteGroupSelected",
      JSON.stringify(group.concat([routeGroup]))
    );

    if (
      obj.hasOwnProperty("Region") &&
      !bcData.some(function (o) {
        return o["title"] === "Region";
      })
    ) {
      let data = obj["Region"];
      let bcContent = obj["Region"];
      let multiSelectLS;
      let regionId;

      if (data.includes(",")) {
        data = `'${data.split(",").join("','")}'`;
      } else if (
        data.charAt(0) !== "'" &&
        data.charAt(data.length - 1) !== "'"
      ) {
        data = `'${data}'`;
      }

      if (
        bcContent.charAt(0) === "'" &&
        bcContent.charAt(bcContent.length - 1) === "'"
      ) {
        regionId = bcContent.substring(1, bcContent.length - 1);
      } else if (bcContent.includes(",")) {
        multiSelectLS = bcContent.split(",");
        regionId = bcContent;
      } else {
        regionId = bcContent;
      }

      bcData.push({ val: regionId, title: "Region" });
      self.setState({ regionId: data });
      let regionLS = bcContent.includes(",")
        ? multiSelectLS
        : region.concat([regionId]);
      window.localStorage.setItem(
        "RouteRegionSelected",
        JSON.stringify(regionLS)
      );
    }
    if (
      obj.hasOwnProperty("Country") &&
      !bcData.some(function (o) {
        return o["title"] === "Country";
      })
    ) {
      let data = obj["Country"];
      let bcContent = obj["Country"];
      let multiSelectLS;
      let countryId;

      if (data.includes(",")) {
        data = `'${data.split(",").join("','")}'`;
      } else if (
        data.charAt(0) !== "'" &&
        data.charAt(data.length - 1) !== "'"
      ) {
        data = `'${data}'`;
      }
      if (
        bcContent.charAt(0) === "'" &&
        bcContent.charAt(bcContent.length - 1) === "'"
      ) {
        countryId = bcContent.substring(1, bcContent.length - 1);
      } else if (bcContent.includes(",")) {
        multiSelectLS = bcContent.split(",");
        countryId = bcContent;
      } else {
        countryId = bcContent;
      }
      bcData.push({ val: countryId, title: "Country" });
      self.setState({ countryId: data });
      let countryLS = bcContent.includes(",")
        ? multiSelectLS
        : country.concat([countryId]);
      window.localStorage.setItem(
        "RouteCountrySelected",
        JSON.stringify(countryLS)
      );
      console.log("rahul Country", countryId, data);
    }
    if (
      obj.hasOwnProperty("Route") &&
      !bcData.some(function (o) {
        return o["title"] === "Route";
      })
    ) {
      let data = obj["Route"];
      let bcContent = obj["Route"];
      let multiSelectLS;
      let routeId;

      if (data.includes(",")) {
        data = `'${data.split(",").join("','")}'`;
      } else if (
        data.charAt(0) !== "'" &&
        data.charAt(data.length - 1) !== "'"
      ) {
        data = `'${data}'`;
      }
      if (
        bcContent.charAt(0) === "'" &&
        bcContent.charAt(bcContent.length - 1) === "'"
      ) {
        routeId = bcContent.substring(1, bcContent.length - 1);
      } else if (bcContent.includes(",")) {
        multiSelectLS = bcContent.split(",");
        routeId = bcContent;
      } else {
        routeId = bcContent;
      }

      bcData.push({ val: routeId, title: "Route" });
      self.setState({ routeId: data });
      let cityLS = bcContent.includes(",")
        ? multiSelectLS
        : city.concat([routeId]);
      window.localStorage.setItem("RouteSelected", JSON.stringify(cityLS));
      console.log("rahul Route", routeId, data);
    }
    if (
      obj.hasOwnProperty("Leg") &&
      !bcData.some(function (o) {
        return o["title"] === "Leg";
      })
    ) {
      bcData.push({ val: obj["Leg"], title: "Leg" });
      console.log("rahul Leg", obj["Leg"]);

      self.setState({ leg: `'${obj["Leg"]}'` });
      window.localStorage.setItem("LegSelected", obj["Leg"]);
    }
    if (
      obj.hasOwnProperty("Flight") &&
      !bcData.some(function (o) {
        return o["title"] === "Flight";
      })
    ) {
      bcData.push({ val: obj["Flight"], title: "Flight" });
      console.log("rahul Flight", obj["Flight"]);

      self.setState({ flight: obj["Flight"] });
      window.localStorage.setItem("FlightSelected", obj["Flight"]);
    }

    console.log("rahul bcData before", bcData, lastIndex);

    if (bcData.length > 0) {
      var removeArrayIndex = bcData.slice(0, lastIndex);
      bcData = removeArrayIndex;
    }
    console.log("rahul bcData after", bcData);

    this.listHandleClick(data[lastIndex], title[lastIndex], "browserBack");
  }

  getFiltersValue = () => {
    bcData = [];
    let routeGroup = window.localStorage.getItem("RouteGroupSelected");
    let RegionSelected = window.localStorage.getItem("RouteRegionSelected");
    let CountrySelected = window.localStorage.getItem("RouteCountrySelected");
    let RouteSelected = window.localStorage.getItem("RouteSelected");
    let rangeValue = JSON.parse(
      window.localStorage.getItem("rangeValueNextYear")
    );
    let getCabinValue = window.localStorage.getItem("CabinSelected");
    let LegSelected = window.localStorage.getItem("LegSelected");
    let FlightSelected = window.localStorage.getItem("FlightSelected");
    let version = window.localStorage.getItem("RPSVersion");
    let RpsVersion = window.localStorage.getItem("RPSVersion");

    if (routeGroup === null || routeGroup === "" || routeGroup === "Null") {
      if (Object.keys(this.userDetails.route_access).length > 0) {
        routeGroup = this.userDetails.route_access["selectedRouteGroup"];
      } else {
        routeGroup = ["Network"];
      }
    } else {
      routeGroup = JSON.parse(routeGroup);
    }

    let cabinSelectedDropDown =
      getCabinValue === null || getCabinValue === "Null"
        ? []
        : JSON.parse(getCabinValue);
    getCabinValue =
      cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : "Null";

    this.setState(
      {
        routeGroup: routeGroup.join("','"),
        regionId:
          RegionSelected === null ||
          RegionSelected === "Null" ||
          RegionSelected === ""
            ? "*"
            : JSON.parse(RegionSelected),
        countryId:
          CountrySelected === null ||
          CountrySelected === "Null" ||
          CountrySelected === ""
            ? "*"
            : JSON.parse(CountrySelected),
        routeId:
          RouteSelected === null ||
          RouteSelected === "Null" ||
          RouteSelected === ""
            ? "*"
            : JSON.parse(RouteSelected),
        leg:
          LegSelected === null || LegSelected === "Null" || LegSelected === ""
            ? "*"
            : `'${LegSelected}'`,
        flight:
          FlightSelected === null ||
          FlightSelected === "Null" ||
          FlightSelected === ""
            ? "*"
            : `'${FlightSelected}'`,
        // gettingMonth: window.monthNumToName(rangeValue.from.month),
        // gettingYear: rangeValue.from.year,
        gettingMonth: window.monthNumToName(1),
        // gettingYear: currentYear + 1,
        getCabinValue: getCabinValue,
        cabinSelectedDropDown: cabinSelectedDropDown,
        version: version === null ? "Null" : 1,
        RpsVersion: RpsVersion,
      },
      () => this.getInitialData()
    );
  };

  getInitialData() {
    let self = this;
    let {
      routeGroup,
      currency,
      gettingMonth,
      gettingYear,
      regionId,
      countryId,
      routeId,
      leg,
      flight,
      getCabinValue,
      ancillaryParam,
      routeParam,
      version,
    } = this.state;

    self.setState({
      loading: true,
      loading2: true,
      firstLoadList: true,
      monthRowData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });

    self.getInitialListData(regionId, countryId, routeId, leg, flight);

    apiServices
      .getRPSRouteMonthTables(
        currency,
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        getCabinValue,
        version
      )
      .then(function (result) {
        self.setState({ loading: false, firstLoadList: false });
        if (result) {
          console.log("rahul month", result);
          var totalData = result[0].totalData;
          var columnName = result[0].columnName;
          var monthRowData = result[0].rowData;

          monthData = monthRowData;
          self.setState({
            monthRowData: self.getHighlightedMonth(
              monthRowData,
              gettingMonth,
              gettingYear
            ),
            monthColumn: columnName,
            monthTotalData: totalData,
            gettingYear: monthRowData[0].Year
          });
        }
        
        if (ancillaryParam) {
      self.getDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        "Ancillary"
      );
      self.setState({ type: "Ancillary" });
    } else if (routeParam) {
      self.getDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        "OD"
        );
        self.setState({ type: "OD" });
      } else {
      self.getDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        "Null"
        );
      }
    });
    }

  getInitialListData = (regionId, countryId, routeId, LEG, FLIGHT) => {
    const self = this;
    const routeAccess = this.userDetails.route_access;
    let leg = LEG.substring(1, LEG.length - 1);
    let flight = FLIGHT.substring(1, FLIGHT.length - 1);

    if (Object.keys(routeAccess).length > 0) {
      self.setState({ accessLevelDisable: true });
    }
    const regionLevelAccess = routeAccess.hasOwnProperty("selectedRouteRegion");
    self.setState({ regionLevelAccess });
    const countryLevelAccess = routeAccess.hasOwnProperty(
      "selectedRouteCountry"
    );
    const routeLevelAccess = routeAccess.hasOwnProperty("selectedRoute");

    if (regionId !== "*") {
      bcData.push({
        val: regionId,
        title: "Region",
        disable: countryLevelAccess,
      });
      self.setState({ selectedData: regionId });
    }
    if (countryId !== "*") {
      bcData.push({
        val: countryId,
        title: "Country",
        disable: routeLevelAccess,
      });
      self.setState({ selectedData: countryId });
    }
    if (routeId !== "*") {
      bcData.push({ val: routeId, title: "Route" });
      self.setState({ selectedData: routeId });
    }
    if (leg !== "*") {
      bcData.push({ val: leg, flight, title: "Leg" });
      self.setState({ selectedData: LEG });
    }
    if (flight !== "*") {
      bcData.push({ val: flight, title: "Flight" });
      self.setState({ selectedData: FLIGHT });
    }
  };

  getMonthDrillDownData = (
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight
  ) => {
    var self = this;
    let { currency, gettingMonth, getCabinValue, type, gettingYear, version } =
      this.state;
    self.setState({
      loading: true,
      loading2: true,
      monthRowData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });

    apiServices
      .getRPSRouteMonthTables(
        currency,
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        getCabinValue,
        version
      )
      .then(function (result) {
        self.setState({ loading: false });
        if (result) {
          var totalData = result[0].totalData;
          var columnName = result[0].columnName;
          var monthRowData = result[0].rowData;
          monthData = monthRowData;
          self.setState({
            monthRowData: self.getHighlightedMonth(
              monthRowData,
              gettingMonth,
              gettingYear
            ),
            monthColumn: columnName,
            monthTotalData: totalData,
          });
        }
      });

    apiServices
      .getRPSRouteDrillDownData(
        gettingYear,
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
      )
      .then((result) => {
        self.setState({ loading2: false });
        if (result) {
          self.setState({
            drillDownTotalData: result[0].totalData,
            drillDownData: result[0].rowData,
            drillDownColumn: result[0].columnName,
            tabName:
              type === "Null" ? result[0].tabName : result[0].firstTabName,
            regionId: result[0].currentAccess.regionId,
            countryId: result[0].currentAccess.countryId,
            routeId: result[0].currentAccess.routeId,
            leg: result[0].currentAccess.leg,
            flight: result[0].currentAccess.flight,
          });
        }
      });
  };

  getDrillDownData = (
    routeGroup,
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    type
  ) => {
    var self = this;
    let { gettingYear, gettingMonth, getCabinValue, currency, version } =
      this.state;

    apiServices
      .getRPSRouteDrillDownData(
        gettingYear,
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
      )
      .then((result) => {
        self.setState({ loading2: false });
        if (result) {
          console.log("rahul drilldown", result);
          self.setState({
            drillDownTotalData: result[0].totalData,
            drillDownData: result[0].rowData,
            drillDownColumn: result[0].columnName,
            tabName:
              type === "Null" ? result[0].tabName : result[0].firstTabName,
            regionId: result[0].currentAccess.regionId,
            countryId: result[0].currentAccess.countryId,
            routeId: result[0].currentAccess.routeId,
            leg: result[0].currentAccess.leg,
            flight: result[0].currentAccess.flight,
          });
        }
      });
  };

  getHighlightedMonth(monthData, month, year) {
    console.log("rahul 1", monthData);
    let monthNumber = window.monthNameToNum(month);
    let data = monthData.filter((data, index) => {
      var monthName = data.Month;
      const selectedMonth = `${window.shortMonthNumToName(
        monthNumber
      )} ${year}`;
      if (selectedMonth === monthName) {
        data.highlightMe = true;
        this.setState({ ensureIndexVisible: index });
      }
      return data;
    });
    return data;
  }

  getHighlightedRow(updatedData, month) {
    let data = updatedData.map((data, index) => {
      let monthName = data.Month;
      if (
        monthName === `▼ Total ${currentYear - 1}` ||
        monthName === `► Total ${currentYear - 1}`
      ) {
        data.highlightMe = true;
      } else if (
        monthName === `▼ Total ${currentYear + 1}` ||
        monthName === `► Total ${currentYear + 1}`
      ) {
        data.highlightMe = true;
      }
      return data;
    });
    return data;
  }

  monthWiseCellClick = (params) => {
    var self = this;
    let {
      routeGroup,
      currency,
      gettingMonth,
      regionId,
      countryId,
      routeId,
      leg,
      flight,
      getCabinValue,
      type,
      gettingYear,
    } = this.state;
    let selectedMonth = params.data.Month;
    var column = params.colDef.field;

    const monthRowData = this.state.monthRowData.map((d) => {
      d.highlightMe = false;
      return d;
    });
    params.api.updateRowData({ update: monthRowData });

    this.setState({
      gettingMonth: params.data.MonthName,
      gettingYear: params.data.Year,
    });
    const range = {
      from: {
        year: params.data.Year,
        month: window.monthNameToNum(params.data.MonthName),
      },
      to: {
        year: params.data.Year,
        month: window.monthNameToNum(params.data.MonthName),
      },
    };
    window.localStorage.setItem("rangeValueNextYear", JSON.stringify(range));

    if (column === "CY_B" && !selectedMonth.includes("Total")) {
      params.event.stopPropagation();
      self.showLoader();
      apiServices
        .getRouteCabinDetails(
          params.data.Year,
          params.data.MonthName,
          routeGroup,
          regionId,
          countryId,
          routeId,
          leg,
          flight,
          getCabinValue
        )
        .then(function (result) {
          self.hideLoader();
          if (result) {
            var columnName = result[0].columnName;
            var cabinData = result[0].cabinData;
            self.setState({
              tableModalVisible: true,
              modalCompartmentData: cabinData,
              modalCompartmentColumn: columnName,
            });
          }
        });
    } else if (column === "Month" && !selectedMonth.includes("Total")) {
      self.setState({
        loading2: true,
        drillDownData: [],
        drillDownTotalData: [],
      });
      self.getDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight,
        type
      );
    }
  };

  regionCellClick = (params) => {
    var self = this;
    let {
      routeGroup,
      regionId,
      countryId,
      routeId,
      leg,
      flight,
      getCabinValue,
    } = this.state;
    var column = params.colDef.field;
    var selectedData = `'${params.data.firstColumnName}'`;
    var selectedDataWQ = params.data.firstColumnName;
    var selectedTitle = params.colDef.headerName;

    let found;
    bcData.map((data, i) =>
      data.title === selectedTitle ? (found = true) : (found = false)
    );

    if (column === "firstColumnName") {
      if (!found) {
        if (selectedTitle !== "Aircraft") {
          this.storeValuesToLS(
            regionId,
            countryId,
            routeId,
            leg,
            flight,
            getCabinValue,
            selectedDataWQ
          );

          self.setState({ selectedData });
          bcData.push({
            val: params.data.firstColumnName,
            title: selectedTitle,
          });
          if (regionId === "*") {
            self.getMonthDrillDownData(
              routeGroup,
              selectedData,
              countryId,
              routeId,
              leg,
              flight
            );
          } else if (countryId === "*") {
            self.getMonthDrillDownData(
              routeGroup,
              regionId,
              selectedData,
              routeId,
              leg,
              flight
            );
          } else if (routeId === "*") {
            self.getMonthDrillDownData(
              routeGroup,
              regionId,
              countryId,
              selectedData,
              leg,
              flight
            );
          } else if (leg === "*") {
            self.getMonthDrillDownData(
              routeGroup,
              regionId,
              countryId,
              routeId,
              selectedData,
              flight
            );
          } else if (flight === "*") {
            self.getMonthDrillDownData(
              routeGroup,
              regionId,
              countryId,
              routeId,
              leg,
              selectedData
            );
          }
        }
      }
    }
  };

  rectifyURLValues(regionId, countryId, routeId, leg) {
    if (Array.isArray(regionId)) {
      this.selectedRegion = regionId.join(",");
    } else if (regionId.includes("','")) {
      this.selectedRegion = regionId.split("','").join(",");
      this.selectedRegion = this.selectedRegion.substring(
        1,
        this.selectedRegion.length - 1
      );
    } else {
      this.selectedRegion = regionId;
      this.selectedRegion = this.selectedRegion.substring(
        1,
        this.selectedRegion.length - 1
      );
    }

    if (Array.isArray(countryId)) {
      this.selectedCountry = countryId.join(",");
    } else if (regionId.includes("','")) {
      this.selectedCountry = countryId.split("','").join(",");
      this.selectedCountry = this.selectedCountry.substring(
        1,
        this.selectedCountry.length - 1
      );
    } else {
      this.selectedCountry = countryId;
      this.selectedCountry = this.selectedCountry.substring(
        1,
        this.selectedCountry.length - 1
      );
    }

    if (Array.isArray(routeId)) {
      this.selectedRoute = routeId.join(",");
    } else if (regionId.includes("','")) {
      this.selectedRoute = routeId.split("','").join(",");
      this.selectedRoute = this.selectedRoute.substring(
        1,
        this.selectedRoute.length - 1
      );
    } else {
      this.selectedRoute = routeId;
      this.selectedRoute = this.selectedRoute.substring(
        1,
        this.selectedRoute.length - 1
      );
    }

    this.selectedLeg = leg;
    this.selectedLeg = this.selectedLeg.substring(
      1,
      this.selectedLeg.length - 1
    );
  }

  storeValuesToLS(
    regionId,
    countryId,
    routeId,
    leg,
    flight,
    getCabinValue,
    data
  ) {
    let region = [];
    let country = [];
    let route = [];
    let cabin = [];

    this.rectifyURLValues(regionId, countryId, routeId, leg);

    if (regionId === "*") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(data)}`
      );
      region.push(data);
      window.localStorage.setItem(
        "RouteRegionSelected",
        JSON.stringify(region)
      );
    } else if (countryId === "*") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${data}`
      );
      country.push(data);
      window.localStorage.setItem(
        "RouteCountrySelected",
        JSON.stringify(country)
      );
    } else if (routeId === "*") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${data}`
      );
      route.push(data);
      window.localStorage.setItem("RouteSelected", JSON.stringify(route));
    } else if (leg === "*") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${this.selectedRoute}&Leg=${data}`
      );
      window.localStorage.setItem("LegSelected", data);
    } else if (flight === "*") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${this.selectedRoute}&Leg=${this.selectedLeg}&Flight=${data}`
      );
      window.localStorage.setItem("FlightSelected", data);
    } else if (getCabinValue === "Null") {
      cabin.push(data);
      window.localStorage.setItem("CabinSelected", JSON.stringify(cabin));
    }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    self.sendEvent(
      "2",
      `clicked on ${selectedType} tab`,
      "route",
      "Route Page"
    );
    let { routeGroup, regionId, countryId, routeId, leg, flight, monthColumn } =
      this.state;
    self.setState({
      type: selectedType,
      loading2: true,
      drillDownData: [],
      drillDownTotalData: [],
    });

    if (outerTab) {
      this.setState({ outerTab });
    } else {
      this.setState({ outerTab: "" });
    }
    self.getDrillDownData(
      routeGroup,
      regionId,
      countryId,
      routeId,
      leg,
      flight,
      selectedType
    );
    this.gridApiMonth.setColumnDefs(monthColumn);
  };

  cabinSelectChange = (e) => {
    e.preventDefault();
    const getCabinValue = e.target.value;

    this.setState(
      {
        getCabinValue: getCabinValue,
        cabinSelectedDropDown: getCabinValue,
      },
      () => {
        window.localStorage.setItem(
          "CabinSelected",
          JSON.stringify(getCabinValue)
        );
      }
    );
  };

  onCabinClose() {
    var self = this;
    self.sendEvent("2", "clicked on Cabin drop down", "/route", "Route Page");
    let { cabinSelectedDropDown } = this.state;

    if (cabinSelectedDropDown.length > 0) {
      this.getDataOnCabinChange();
    } else {
      this.setState({ getCabinValue: "Null" }, () =>
        this.getDataOnCabinChange()
      );
      window.localStorage.setItem("CabinSelected", "Null");
    }
  }

  getDataOnCabinChange() {
    var self = this;
    self.setState({
      loading: true,
      loading2: true,
      monthRowData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });
    let { routeGroup, regionId, countryId, routeId, leg, flight } = this.state;

    self.getMonthDrillDownData(
      routeGroup,
      regionId,
      countryId,
      routeId,
      leg,
      flight
    );
  }

  homeHandleClick = (e) => {
    var self = this;
    let { routeGroup } = this.state;
    self.setState({
      loading: true,
      loading2: true,
      firstHome: false,
      monthRowData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
      odData: [],
      segmentData: [],
      currency: "bc",
    });

    window.localStorage.setItem("RouteRegionSelected", "Null");
    window.localStorage.setItem("RouteCountrySelected", "Null");
    window.localStorage.setItem("RouteSelected", "Null");
    window.localStorage.setItem("LegSelected", "Null");
    window.localStorage.setItem("FlightSelected", "Null");

    self.getMonthDrillDownData(routeGroup, "*", "*", "*", "*", "*");
    bcData = [];
    this.props.history.push(`${this.pathName}?RouteGroup=${routeGroup}`);
  };

  listHandleClick = (data, title, selection) => {
    var self = this;
    let { routeGroup, regionId, countryId, routeId, leg, flight } = this.state;
    var selectedData = data;
    if (
      selectedData.charAt(0) !== "'" &&
      selectedData.charAt(selectedData.length - 1) !== "'"
    ) {
      selectedData = `'${data}'`;
    }
    if (data.includes(",")) {
      selectedData = `'${data.split(",").join("','")}'`;
    }
    self.setState({
      selectedData,
      loading: true,
      loading2: true,
      monthRowData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });
    var getColName = decodeURIComponent(title);

    if (selection === "List") {
      var indexEnd = bcData.findIndex(function (d) {
        return d.title == title;
      });
      var removeArrayIndex = bcData.slice(0, indexEnd + 1);
      bcData = removeArrayIndex;

      this.changeURLOnListClick(
        regionId,
        countryId,
        routeId,
        leg,
        data,
        getColName
      );
    } else if (selection === "browserBack") {
      this.onBackPressClearLS(getColName);
    }

    if (getColName === "Region") {
      self.getMonthDrillDownData(routeGroup, selectedData, "*", "*", "*", "*");
    } else if (getColName === "Country") {
      self.getMonthDrillDownData(
        routeGroup,
        regionId,
        selectedData,
        "*",
        "*",
        "*"
      );
    } else if (getColName === "Route") {
      self.getMonthDrillDownData(
        routeGroup,
        regionId,
        countryId,
        selectedData,
        "*",
        "*"
      );
    } else if (getColName === "Leg") {
      self.getMonthDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        selectedData,
        "*"
      );
    } else if (getColName === "Flight") {
      self.getMonthDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        selectedData
      );
    } else if (getColName === "RouteGroup") {
      self.setState({ routeGroup: data }, () => this.homeHandleClick());
    }
  };

  changeURLOnListClick(
    regionId,
    countryId,
    routeId,
    leg,
    selectedData,
    getColName
  ) {
    this.rectifyURLValues(regionId, countryId, routeId, leg);

    if (getColName === "Region") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(selectedData)}`
      );
      window.localStorage.setItem("RouteCountrySelected", "Null");
      window.localStorage.setItem("RouteSelected", "Null");
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Country") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${selectedData}`
      );
      window.localStorage.setItem("RouteSelected", "Null");
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Route") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${selectedData}`
      );
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Leg") {
      window.localStorage.setItem("FlightSelected", "Null");
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${this.selectedRoute}&Leg=${selectedData}`
      );
    } else if (getColName === "Flight") {
      this.props.history.push(
        `${this.pathName}?RouteGroup=${
          this.state.routeGroup
        }&Region=${encodeURIComponent(this.selectedRegion)}&Country=${
          this.selectedCountry
        }&Route=${this.selectedRoute}&Leg=${
          this.selectedLeg
        }&Flight=${selectedData}`
      );
    }
  }

  onBackPressClearLS(getColName) {
    if (getColName === "Region") {
      window.localStorage.setItem("RouteCountrySelected", "Null");
      window.localStorage.setItem("RouteSelected", "Null");
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Country") {
      window.localStorage.setItem("RouteSelected", "Null");
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Route") {
      window.localStorage.setItem("LegSelected", "Null");
      window.localStorage.setItem("FlightSelected", "Null");
    } else if (getColName === "Leg") {
      window.localStorage.setItem("FlightSelected", "Null");
    }
  }

  currency = (e) => {
    let currency = e.target.value;
    const { routeGroup, regionId, countryId, routeId, leg, flight } =
      this.state;
    this.setState({ currency: currency }, () =>
      this.getMonthDrillDownData(
        routeGroup,
        regionId,
        countryId,
        routeId,
        leg,
        flight
      )
    );
  };

  closeChartModal() {
    this.setState({ chartVisible: false });
  }

  closeTableModal() {
    this.setState({ tableModalVisible: false });
  }

  redirection = (e) => {
    this.sendEvent(
      "2",
      "clicked on POS/Route drop down",
      "/route",
      "Route Page"
    );
    let name = e.target.value;

    if (name === "POS") {
      this.props.history.push(`/rpsPos${Constant.getPOSFiltersSearchURL()}`);
      bcData = [];
    } else {
      this.props.history.push("/route");
      bcData = [];
    }
  };

  callAccess(routeGroup) {
    let routeGroupArray = [];
    routeGroupArray.push(routeGroup);
    window.localStorage.setItem(
      "RouteGroupSelected",
      JSON.stringify(routeGroupArray)
    );
    this.setState({ routeGroup }, () => this.homeHandleClick());
  }

  showLoader = () => {
    $("#loaderImage").addClass("loader-visible");
  };

  hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible");
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
  };

  gotoFirstPage = () => {
    const { totalPages, paginationSize, totalRecords } = this.state;
    const remainder = totalRecords % paginationSize;
    const pageEnd =
      remainder < paginationSize && totalRecords < paginationSize
        ? remainder
        : paginationSize;
    this.setState(
      {
        count: 1,
        paginationStart: 1,
        paginationEnd: pageEnd,
      },
      () => {
        this.paginationClick();
      }
    );
  };

  gotoLastPage = () => {
    const { totalPages, paginationSize, totalRecords } = this.state;
    const startDigit = paginationSize * (totalPages - 1);
    console.log("Updating paginationEnd gotoLstPage");
    this.setState(
      {
        count: totalPages,
        paginationStart: startDigit + 1,
        paginationEnd: totalRecords,
      },
      () => this.paginationClick()
    );
  };

  gotoPreviousPage = () => {
    const {
      count,
      currentPage,
      totalPages,
      paginationSize,
      paginationStart,
      paginationEnd,
      totalRecords,
    } = this.state;
    const remainder = totalRecords % paginationSize;
    const fromLast = currentPage === totalPages;
    const decrement = fromLast && remainder > 0 ? remainder : paginationSize;
    this.setState(
      {
        count: count - 1,
        paginationStart: paginationStart - paginationSize,
        paginationEnd: paginationEnd - decrement,
      },
      () => this.paginationClick()
    );
  };

  paginationClick = () => {
    // if (this.state.toggleChange) {
    //   this.getSwappedTopMarketData();
    // } else {
    this.setState({ odData: "" }, () => this.posContributionClick());
    // }
  };

  gotoNextPage = () => {
    const {
      count,
      currentPage,
      totalPages,
      paginationSize,
      paginationStart,
      paginationEnd,
      totalRecords,
    } = this.state;
    const remainder = totalRecords % paginationSize;
    const tolast = currentPage === totalPages - 1;
    const increment = tolast && remainder > 0 ? remainder : paginationSize;
    this.setState(
      {
        count: count + 1,
        paginationStart: paginationStart + paginationSize,
        paginationEnd: paginationEnd + increment,
      },
      () => this.paginationClick()
    );
  };

  renderTabs() {
    let {
      tabName,
      gettingMonth,
      regionId,
      countryId,
      routeId,
      leg,
      flight,
      getCabinValue,
      routeGroup,
      gettingYear,
      type,
      outerTab,
      ancillaryParam,
      routeParam,
    } = this.state;
    // const downloadURLDrillDown = apiServices.exportCSVRouteDrillDownURL(gettingYear, gettingMonth, routeGroup, regionId, countryId, routeId, leg, flight, getCabinValue, type)
    // const downloadURLMonthly = apiServices.exportCSVRouteMonthlyURL(routeGroup, regionId, countryId, routeId, leg, flight, getCabinValue)

    return (
      <ul className="nav nav-tabs" role="tablist">
        {tabName === "Flight" || tabName === "Leg" || tabName === "Aircraft" ? (
          <li
            role="presentation"
            className={`${routeParam ? "active" : ""}`}
            onClick={() => this.tabClick("OD", "Route")}
          >
            <a
              href="#Section5"
              aria-controls="profile"
              role="tab"
              data-toggle="tab"
            >
              Route
            </a>
          </li>
        ) : (
          ""
        )}

        {/* { tabName === 'Flight' ?
          <li role="presentation" onClick={this.tabClick}>
            <a href="#Section7" aria-controls="messages" role="tab" data-toggle="tab">
              Leg
            </a>
          </li> : ''} */}

        {tabName === "Aircraft" ? (
          <li
            role="presentation"
            onClick={() => this.tabClick("Flights", "Flight")}
          >
            <a
              href="#Section7"
              aria-controls="messages"
              role="tab"
              data-toggle="tab"
            >
              Flight
            </a>
          </li>
        ) : (
          ""
        )}

        <li
          id="regionTab"
          role="presentation"
          className={`${ancillaryParam ? "" : routeParam ? "" : "active"}`}
          onClick={() => this.tabClick("Null")}
        >
          <a
            href="#Section2"
            aria-controls="profile"
            role="tab"
            data-toggle="tab"
          >
            {tabName}
          </a>
        </li>

        {tabName === "Route" ? (
          ""
        ) : tabName === "Flight" ? (
          ""
        ) : tabName === "Aircraft" ? (
          ""
        ) : tabName === "Leg" ? (
          ""
        ) : (
          <li
            role="presentation"
            className={`${
              routeParam ? "active" : outerTab === "Route" ? "active" : ""
            }`}
            onClick={() => this.tabClick("OD")}
          >
            <a
              href="#Section5"
              aria-controls="profile"
              role="tab"
              data-toggle="tab"
            >
              Route
            </a>
          </li>
        )}

        {/* {tabName === 'Cabin' ? '' : tabName === 'Leg' ? '' : tabName === 'Flight' ? '' :
          <li role="presentation" onClick={this.legTabClick}>
            <a href="#Section7" aria-controls="messages" role="tab" data-toggle="tab">
              Leg
             </a>
          </li>} */}

        {tabName === "Flight" ? (
          ""
        ) : tabName === "Aircraft" ? (
          ""
        ) : (
          <li
            role="presentation"
            onClick={() => this.tabClick("Flights")}
            className={`${outerTab === "Flight" ? "active" : ""}`}
          >
            <a
              href="#Section7"
              aria-controls="messages"
              role="tab"
              data-toggle="tab"
            >
              Flight
            </a>
          </li>
        )}

        {tabName === "Aircraft" ? (
          ""
        ) : (
          <li
            role="presentation"
            onClick={() => this.tabClick("Aircraft")}
            className={`${outerTab === "Aircraft" ? "active" : ""}`}
          >
            <a
              href="#Section8"
              aria-controls="messages"
              role="tab"
              data-toggle="tab"
            >
              Aircraft
            </a>
          </li>
        )}

        {
          <li role="presentation" onClick={() => this.tabClick("Cabin")}>
            <a
              href="#Section9"
              aria-controls="messages"
              role="tab"
              data-toggle="tab"
            >
              Cabin
            </a>
          </li>
        }

        {/* <li role="presentation" onClick={this.rbdTabClick}>
          <a href="#Section4" aria-controls="messages" role="tab" data-toggle="tab">
            RBD
            </a>
        </li> */}

        {/* <DownloadCSV url={downloadURLDrillDown} name={`Route DRILLDOWN`} path={`/route`} page={`Route Page`} />
        <DownloadCSV url={downloadURLMonthly} name={`Route MONTHLY`} path={`/route`} page={`Route Page`} />
        {routeId !== '*' ? <button className='btn download' onClick={this.posContributionClick}>POS Contribution</button> : ''} */}
      </ul>
    );
  }

  onChangePosContri = (e) => {
    e.stopPropagation();
    this.setState(
      {
        posContributionTable: e.target.value,
        currentPage: "",
        totalPages: "",
        totalRecords: "",
        paginationStart: 1,
        paginationEnd: "",
        paginationSize: "",
        count: 1,
      },
      () => {
        this.posContributionClick();
      }
    );
  };

  renderPosContributionModal() {
    return (
      <Modal
        show={this.state.posContributionModal}
        onHide={() =>
          this.setState({
            posContributionModal: false,
            posContributionTable: "OD",
            currentPage: "",
            totalPages: "",
            totalRecords: "",
            paginationStart: 1,
            paginationEnd: "",
            paginationSize: "",
            count: 1,
          })
        }
        aria-labelledby="ModalHeader"
        className="posContri"
      >
        <Modal.Header closeButton>
          <Modal.Title
            id="ModalHeader"
            title={`POS Contribution for ${Constant.removeQuotes(
              this.state.routeId
            )}`}
          >{`POS Contribution for ${Constant.removeQuotes(
            this.state.routeId
          )}`}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div style={{ width: "100%" }}>
            <div className="posContriBodyTop">
              <h3 className="headingPOS">
                {this.state.posContributionTableHeader}
              </h3>
              <select
                className="form-control cabinselect pos-route-dropdown"
                onChange={(e) => this.onChangePosContri(e)}
              >
                <option value="OD" selected={true}>
                  OD Wise
                </option>
                <option value="Segment">Segment Wise</option>
              </select>
            </div>
            <DataTableComponent
              rowData={this.state.odData}
              columnDefs={this.state.odColumn}
              route={true}
            />
          </div>
          <Pagination
            paginationStart={this.state.paginationStart}
            paginationEnd={this.state.paginationEnd}
            totalRecords={this.state.totalRecords}
            currentPage={this.state.currentPage}
            TotalPages={this.state.totalPages}
            gotoFirstPage={() => this.gotoFirstPage()}
            gotoLastPage={() => this.gotoLastPage()}
            gotoPreviousPage={() => this.gotoPreviousPage()}
            gotoNexttPage={() => this.gotoNextPage()}
          />
          {/* <div style={{ width: '100%' }}>
            <h3 className='headingPOS'>Segment Wise</h3>
            <DataTableComponent
              rowData={this.state.segmentData}
              columnDefs={this.state.segmentColumn}
              route={true}
            />
          </div> */}
        </Modal.Body>
      </Modal>
    );
  }

  gridApiMonthly = (api) => {
    this.gridApiMonth = api;
  };

  render() {
    const downloadURL = localStorage.getItem("RPSRouteDownloadURL");
    const {
      ancillaryParam,
      routeParam,
      cabinOption,
      cabinSelectedDropDown,
      cabinDisable,
      routeGroup,
      accessLevelDisable,
      firstLoadList,
      regionLevelAccess,
    } = this.state;

    return (
      <div className="rps">
        <TopMenuBar dashboardPath={`/posDashboard`} {...this.props} />
        {/* <TopMenuBar dashboardPath={`/routeRevenuePlanning`} {...this.props} /> */}
        <Loader />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: "0px" }}>
              <div className="col-md-7 col-sm-7 col-xs-12 toggle1">
                <select
                  className="form-control cabinselect pos-route-dropdown"
                  onChange={(e) => this.redirection(e)}
                >
                  <option value="POS">POS RPS</option>
                  <option value="Route" selected={true}>
                    Route RPS
                  </option>
                </select>
                {firstLoadList ? (
                  ""
                ) : (
                  <div className="route-access">
                    {routeGroup}
                    <div className="triangle-up"></div>
                    <div className="route-groups">
                      <div
                        className={`route-main ${
                          accessLevelDisable ? " route-main-disable" : ""
                        }`}
                      >
                        <span
                          className={`${
                            accessLevelDisable ? " route-access-disable" : ""
                          }`}
                          onClick={() => this.callAccess("Network")}
                        >
                          Network
                        </span>
                      </div>
                      <div
                        className={`route-main ${
                          accessLevelDisable
                            ? regionLevelAccess
                              ? "route-main-disable"
                              : routeGroup === "Domestic"
                              ? ""
                              : "route-main-disable"
                            : ""
                        }`}
                      >
                        <span
                          className={`${
                            accessLevelDisable
                              ? regionLevelAccess
                                ? "route-access-disable"
                                : routeGroup === "Domestic"
                                ? ""
                                : "route-access-disable"
                              : ""
                          }`}
                          onClick={() => this.callAccess("Domestic")}
                        >
                          Domestic
                        </span>
                      </div>
                      <div
                        className={`route-main international ${
                          accessLevelDisable
                            ? regionLevelAccess
                              ? "route-main-disable"
                              : routeGroup === "International"
                              ? ""
                              : "route-main-disable"
                            : ""
                        }`}
                      >
                        <span
                          className={`${
                            accessLevelDisable
                              ? regionLevelAccess
                                ? "route-access-disable"
                                : routeGroup === "International"
                                ? ""
                                : "route-access-disable"
                              : ""
                          }`}
                          onClick={() => this.callAccess("International")}
                        >
                          International
                        </span>
                      </div>
                    </div>
                  </div>
                )}
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      {this.state.firstLoadList
                        ? ""
                        : bcData.map((item) => (
                            <div
                              style={{
                                cursor: item.disable
                                  ? "not-allowed"
                                  : "pointer",
                              }}
                            >
                              <li
                                className={`${
                                  item.disable ? "breadcrumb-disable" : ""
                                }`}
                                onClick={(e) =>
                                  this.listHandleClick(
                                    e.target.id,
                                    item.title,
                                    "List"
                                  )
                                }
                                id={item.val}
                                title={`${item.title} : ${item.val}`}
                              >
                                {` > ${item.val}`}
                              </li>
                            </div>
                          ))}

                      {this.state.version === 0 ||
                      this.state.version === "null" ||
                      (this.state.version === "Null" &&
                        this.state.regionId === "*" &&
                        this.state.countryId === "*") ? (
                        <div>
                          <h2 style={{ marginLeft: "20px" }}>
                            Demand Estimation Screen
                          </h2>
                        </div>
                      ) : (
                        <div>
                          <h2 style={{ marginLeft: "20px" }}>
                            Network Optimal Target Screen
                          </h2>
                        </div>
                      )}

                      {/* {this.state.regionId === "*" &&
                      this.state.countryId === "*" &&
                      this.state.version != 0 ? (
                        <div>
                          <h2 style={{ marginLeft: "20px" }}>
                            Network Optimal Target Screen
                          </h2>
                        </div>
                      ) : (
                        ""
                      )} */}
                    </ol>
                  </nav>
                </section>
              </div>

              <div
                className="col-md-5 col-sm-5 col-xs-12 toggle2"
                style={{ marginRight: "20px" }}
              >
                {console.log(
                  this.state.version.length,
                  this.state.version,
                  ":::00000::::"
                )}
                {this.state.version === 0 ||
                this.state.version === "null" ||
                this.state.version === "Null" ? (
                  ""
                ) : (
                  <h4 style={{ marginRight: "10px" }}>
                    {" "}
                    Version : {this.state.RpsVersion}
                  </h4>
                )}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <DownloadCSV
                    url={downloadURL}
                    name={`RPS POS`}
                    path={`/rpsPos`}
                    page
                  />
                </div>
                {/* <div className='cabin-selection'>
                  <h4>Select Cabin :</h4>
                  <FormControl className="select-group">
                    <InputLabel id="demo-mutiple-checkbox-label">All</InputLabel>
                    <Select
                      labelId="demo-mutiple-checkbox-label"
                      className={`${cabinDisable ? 'disable' : ''}`}
                      id={`demo-mutiple-checkbox`}
                      multiple
                      value={cabinSelectedDropDown}
                      onChange={(e) => this.cabinSelectChange(e)}
                      input={<Input />}
                      renderValue={selected => {
                        return selected.join(',')
                      }}
                      onClose={() => this.onCabinClose()}
                      MenuProps={{ classes: 'disable' }}
                    >
                      {cabinOption.map(item => (
                        <MenuItem key={item.ClassValue} value={item.ClassValue}>
                          <Checkbox checked={cabinSelectedDropDown.indexOf(item.ClassText) > -1} />
                          <ListItemText primary={item.ClassText} />
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </div> */}
                {/* <h4>Select Currency :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.currency(e)} disabled={this.state.countryId === '*' ? true : false}>
                  <option value='bc' selected={this.state.countryId === '*' || this.state.toggle === 'bc' ? true : false}>BC</option>
                  <option value='lc'>LC</option>
                </select> */}
              </div>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div
              className="x_panel"
              style={{ marginTop: "10px", height: "calc(100vh - 130px)" }}
            >
              <div className="x_content">
                <DataTableComponent
                  rowData={this.state.monthRowData}
                  columnDefs={this.state.monthColumn}
                  onCellClicked={(cellData) =>
                    this.monthWiseCellClick(cellData)
                  }
                  frameworkComponents={{
                    customHeaderGroupComponent:
                      RPSRouteDownloadCustomHeaderGroup,
                  }}
                  loading={this.state.loading}
                  rowClassRules={this.state.routeMonthRowClassRule}
                  route={true}
                  ensureIndexVisible={this.state.ensureIndexVisible}
                  gridApi={this.gridApiMonthly}
                />
                <TotalRow
                  rowData={this.state.monthTotalData}
                  columnDefs={this.state.monthColumn}
                  frameworkComponents={{
                    customHeaderGroupComponent:
                      RPSRouteDownloadCustomHeaderGroup,
                  }}
                  loading={this.state.loading}
                  responsive={true}
                  reducingPadding={true}
                />

                <div
                  className="tab"
                  id="posTableTab"
                  role="tabpanel"
                  style={{ marginTop: "10px" }}
                >
                  {this.renderTabs()}

                  <div className="tab-content tabs">
                    <div
                      role="tabpanel"
                      className={`tab-pane fade in ${
                        ancillaryParam ? "" : routeParam ? "" : "active"
                      }`}
                      id="Section2"
                    >
                      {/* Region */}
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) =>
                          this.regionCellClick(cellData)
                        }
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* RBD */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section4"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* Route */}
                    <div
                      role="tabpanel"
                      className={`tab-pane fade in ${
                        routeParam ? "active" : ""
                      }`}
                      id="Section5"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* Leg */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section7"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* Flights */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section7"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* Aircarft */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section8"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    {/* Compartment */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section9"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        route={true}
                      />
                      <TotalRow
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <DatatableModelDetails
            tableModalVisible={this.state.tableModalVisible}
            rowData={this.state.modalCompartmentData}
            columns={this.state.modalCompartmentColumn}
            header={`${this.state.gettingMonth} ${this.state.gettingYear}`}
          />
          <ChartModelDetails
            chartVisible={this.state.chartVisible}
            datas={this.state.modelRegionDatas}
            columns={this.state.modelregioncolumn}
            closeChartModal={() => this.closeChartModal()}
          />
          {this.renderPosContributionModal()}
        </div>
      </div>
    );
  }
}

const NewComponentRoute = BrowserToProps(RPSRoute);

export default NewComponentRoute;
