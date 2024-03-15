import React, { Component, version } from "react";
import Swal from "sweetalert2";
import Box from "@mui/material/Box";
import Slider from "@mui/material/Slider";
import { Typography } from "@mui/material";
import BrowserToProps from "react-browser-to-props";
import Modal from "react-bootstrap-modal";
import APIServices from "../../API/apiservices";
import api from "../../API/api";
import eventApi from "../../API/eventApi";
import ChartModelDetails from "../../Component/chartModel";
import DatatableModelDetails from "../../Component/dataTableModel";
import Loader from "../../Component/Loader";
import DataTableComponent from "../../Component/DataTableComponent";
import TotalRow from "../../Component/TotalRow";
import NotificationRightSidebar from "../../Component/NotificationRightSidebar";
import { string } from "../../Constants/string";
import Constant from "../../Constants/validator";
import color from "../../Constants/color";
import cookieStorage from "../../Constants/cookie-storage";
import $ from "jquery";
import "../../App";
import "./RPS.scss";
import { tree } from "d3";
import TopMenuBar from "../../Component/TopMenuBar";
import RPSPosDownloadCustomHeaderGroup from "./RPSPosDownloadCustomHeaderGroup";
import axios from "axios";
import config from "../../Constants/config";
import { values } from "lodash";
import DownloadCSV from "../../Component/DownloadCSV";

const apiServices = new APIServices();
const currentYear = new Date().getFullYear();
let bcData = [];
let originalMonthData = [];
let originalMonthTotalData = [];
let originalDrilldownData = [];
let originalDrilldownTotalData = [];

const API_URL = config.API_URL;

class RPSPos extends Component {
  constructor(props) {
    super(props);
    this.pathName = window.location.pathname;
    this.selectedRegion = null;
    this.selectedCountry = null;
    this.selectedCity = null;
    // this.selectedOd = "null";
    this.gridApiMonth = null;
    this.userData = Constant.loggedinUser(
      JSON.parse(cookieStorage.getCookie("userDetails"))
    );
    // this.isPOSNetworkAdmin = userData.isPOSNetworkAdmin ? (!userData.isPOSNetworkAdmin).toString() : `!${userData.isPOSNetworkAdmin.toString()}`
    this.state = {
      monthData: [],
      monthTotalData: [],
      monthcolumns: [],
      monthcolumnsTotal: [],
      modalData: [],
      drillDownColumn: [],
      drillDownData: [],
      modeldrillDownDatas: [],
      modeldrillDownColumn: [],
      tableDatas: true,
      gettingMonth: null,
      gettingYear: null,
      cabinOption: [],
      getCabinValue: [],
      cabinSelectedDropDown: [],
      cabinDisable: true,
      toggle: "bc",
      tabName: "Region",
      regionId: "*",
      countryId: "*",
      cityId: "*",
      commonOD: "*",
      type: "Null",
      baseAccess: "",
      selectedData: "Null",
      infareData: [],
      infareModalVisible: false,
      infareGraphHeading: "",
      loading: true,
      loading2: true,
      loading3: false,
      firstLoadList: false,
      showLastYearRows: false,
      showNextYearRows: false,
      accessLevelDisable: false,
      firstHome: true,
      infareCurrency: "",
      outerTab: false,
      ancillary: false,
      ensureIndexVisible: null,
      LoadFactorValue: 90,
      monthRowClassRule: {
        "highlight-row": "data.highlightMe",
        "ag-row-disableChkbox": `${this.userData.accessLevelPOS.toString()} <= data.Approval_level`,
        "approved-row": `${this.userData.accessLevelPOS.toString()} <= data.Approval_level`,
        "rejected-row": `${this.userData.accessLevelPOS.toString()} <= data.Reject_level`,
      },
      displayModal: false,
      displayModal2: false,
      selectedMonths: [],
      action: "",
      isEdited: false,
      isSubmitted: false,
      editedField: "",
      editedRowIndex: null,
      editedvalue: null,
      isActionPerformed: false,
      isAllApproved: false,
      versionList: [],
      selectedVersion: "Null",
      selectedNewVersion: "Null",
      version_action: "",
      count: 1,
      messageArray: [],
      msgResponseData: [],
      msgCount: 0,
      msgCountVisible: false,
      hasMore: true,
      btndisable: false,
      disableRouteConversion: false,
      disableOptimization: false,
      disableOptimzedRouteConversion: false,
      odsearchvalue: "",
      rowDataforChannel: "",
    };
    this.sendEvent("1", "viewed Pos Page", "pos", "Pos Page");
  }

  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name} Page`,
    };
    // eventApi.sendEvent(eventData)
  };

  componentDidMount() {
    this.localStorageValues();
  }

  componentWillMount() {
    var self = this;
    const { CYear } = this.props;
    self.getFiltersValue();
    self.getMessages();
    apiServices.getClassNameDetails().then((result) => {
      if (result) {
        var classData = result[0].classDatas;
        self.setState({ cabinOption: classData });
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

  localStorageValues = () => {
    var rpsValues = JSON.parse(
      window.localStorage.getItem("RPSAdminScreenValues")
    );
    console.log(rpsValues, "valuestwo");
    if (rpsValues == null) {
      this.flownValue = "Null";
      this.flownValueLength = "Null";
      this.Cyear = "Null";
      this.Tyear = "Null";
      this.Channelyear = "Null";
      this.Segmentationyear = "Null";
    } else {
      this.flownValue = rpsValues.FlownValue;
      this.flownValueLength = rpsValues.FlownValueLength;
      this.Cyear = rpsValues.CYearValue;
      this.Tyear = rpsValues.TYearValue;
      this.Channelyear = rpsValues.ChannelYearValue;
      this.Segmentationyear = rpsValues.SegmentationYearValue;
    }
  };

  getDefaultHeader = () => {
    const token = cookieStorage.getCookie("Authorization");
    return {
      headers: {
        Authorization: token,
      },
    };
  };

  pushURLToBcData(obj, title, data, lastIndex) {
    const self = this;
    let region = [];
    let country = [];
    let city = [];

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
      }
      // else if(bcContent.includes("','")){
      //   multiSelectLS = bcContent.split("','");
      //   regionId = bcContent;
      // }
      else {
        regionId = bcContent;
      }
      console.log("rahul Region", multiSelectLS);
      bcData.push({ val: regionId, title: "Region" });
      self.setState({ regionId: data });
      let regionLS = bcContent.includes(",")
        ? multiSelectLS
        : region.concat([regionId]);
      window.localStorage.setItem("RegionSelected", JSON.stringify(regionLS));
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
      window.localStorage.setItem("CountrySelected", JSON.stringify(countryLS));
      console.log("rahul Country", countryId, data);
    }
    if (
      obj.hasOwnProperty("POS") &&
      !bcData.some(function (o) {
        return o["title"] === "POS";
      })
    ) {
      let data = obj["POS"];
      let bcContent = obj["POS"];
      let multiSelectLS;
      let cityId;

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
        cityId = bcContent.substring(1, bcContent.length - 1);
      } else if (bcContent.includes(",")) {
        multiSelectLS = bcContent.split(",");
        cityId = bcContent;
      } else {
        cityId = bcContent;
      }

      bcData.push({ val: cityId, title: "POS" });
      self.setState({ cityId: data });
      let cityLS = bcContent.includes(",")
        ? multiSelectLS
        : city.concat([cityId]);
      window.localStorage.setItem("CitySelected", JSON.stringify(cityLS));
      console.log("rahul POS", cityId, data);
    }
    if (
      obj.hasOwnProperty("O%26D") &&
      !bcData.some(function (o) {
        return o["title"] === "O&D";
      })
    ) {
      bcData.push({ val: obj["O%26D"], title: "O&D" });
      console.log("rahul OD", obj["O%26D"]);

      self.setState({ commonOD: obj["O%26D"] });
      window.localStorage.setItem("ODSelected", obj["O%26D"]);
    }

    console.log("rahul bcData before", bcData, lastIndex, data[lastIndex], obj);
    if (bcData.length > 0) {
      var removeArrayIndex = bcData.slice(0, lastIndex + 1);
      bcData = removeArrayIndex;
    }

    this.listHandleClick(data[lastIndex], title[lastIndex], "browserBack");
  }

  getFiltersValue = () => {
    bcData = [];
    let RegionSelected = window.localStorage.getItem("RegionSelected");
    let CountrySelected = window.localStorage.getItem("CountrySelected");
    let CitySelected = window.localStorage.getItem("CitySelected");
    let rangeValue = JSON.parse(
      window.localStorage.getItem("rangeValueNextYear")
    );
    let getCabinValue = window.localStorage.getItem("CabinSelected");
    let ODSelected = window.localStorage.getItem("ODSelected");

    let cabinSelectedDropDown =
      getCabinValue === null || getCabinValue === "Null"
        ? []
        : JSON.parse(getCabinValue);
    getCabinValue =
      cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : "Null";

    CitySelected =
      CitySelected === null || CitySelected === "Null" || CitySelected === ""
        ? "*"
        : JSON.parse(CitySelected);

    this.setState(
      {
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
        cityId: CitySelected,
        commonOD:
          ODSelected === null ||
          ODSelected === "Null" ||
          ODSelected === "" ||
          CitySelected === "*"
            ? "*"
            : `'${ODSelected}'`,
        // gettingMonth: window.monthNumToName(rangeValue.from.month),
        // gettingYear: rangeValue.from.year,
        gettingMonth: window.monthNumToName(1),
        gettingYear: currentYear + 1,
        getCabinValue: getCabinValue,
        cabinSelectedDropDown: cabinSelectedDropDown,
      },
      () => this.getVersionList()
    );
  };

  getVersionList(afterFreeze) {
    api
      .get(`freezeandunfreeze`, "hideloader")
      .then((res) => {
        if (res.data.response) {
          const responseData = res.data.response;
          if (responseData.length > 0) {
            const length = responseData.length;
            const selectedVersion = responseData[length - 1].Version;
            console.log(selectedVersion, length, ":::HELLLLLLL:::::");
            const version_action = responseData[length - 1].Version_Status;
            this.setState(
              {
                versionList: responseData,
                selectedVersion: selectedVersion,
                version_action: version_action,
              },
              () => this.getInitialData()
            );
            window.localStorage.setItem("RPSVersion", selectedVersion);
          } else {
            console.log("GGGGGGG");
            this.getInitialData();
          }
        }
      })
      .catch((err) => {
        console.log("rahul message err", err);
      });
  }

  getInitialData = () => {
    var self = this;
    let {
      gettingMonth,
      gettingYear,
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      selectedVersion,
    } = this.state;
    console.log(selectedVersion, ":::VERSION::");
    self.setState({
      loading: true,
      loading2: true,
      firstLoadList: true,
      monthData: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });

    self.getInitialListData(regionId, countryId, cityId, commonOD);

    apiServices
      .getRPSMonthTables(
        regionId,
        countryId,
        cityId,
        commonOD,
        getCabinValue,
        selectedVersion
      )
      .then(function (result) {
        self.setState({ loading: false, firstLoadList: false });
        if (result) {
          var totalData = result[0].totalData;
          var columnName = result[0].columnName;
          var columnNameTotal = result[0].columnNameTotal;
          var rowData = result[0].rowData;

          window.localStorage.setItem(
            "RPS Monthly Data",
            JSON.stringify(rowData)
          );
          originalMonthData = rowData;
          originalMonthTotalData = totalData;

          self.setState({
            monthData: self.getHighlightedMonth(
              rowData,
              gettingMonth,
              gettingYear
            ),
            monthcolumns: columnName,
            monthcolumnsTotal: columnNameTotal,
            monthTotalData: totalData,
            apiMonthlyData: result[0].apiMonthlyData,
            isActionPerformed: result[0].isActionPerformed,
            isAllApproved: result[0].isAllApproved,
            gettingYear: rowData[0].Year
          });
        }

        self.getDrillDownData(regionId, countryId, cityId, commonOD, "Null");
      });
  };

  getInitialListData = (regionId, countryId, cityId, OD) => {
    const self = this;
    const userDetails = JSON.parse(cookieStorage.getCookie("userDetails"));
    let commonOD = OD.substring(1, OD.length - 1);
    let access = userDetails.access;
    let country = "*";
    let city = "*";

    if (access !== "#*") {
      self.setState({ accessLevelDisable: true });
      let accessList = access.split("#");
      country = accessList[2];
      city = accessList[2] === "*" ? "*" : accessList[3];
    }

    if (regionId !== "*") {
      bcData.push({
        val: regionId,
        title: "Region",
        disable: country !== "*" ? true : false,
      });
      self.setState({ selectedData: regionId });
    }
    if (countryId !== "*") {
      bcData.push({
        val: countryId,
        title: "Country",
        disable: city !== "*" ? true : false,
      });
      self.setState({ selectedData: countryId });
    }
    if (cityId !== "*") {
      bcData.push({ val: cityId, title: "POS" });
      self.setState({ selectedData: cityId });
    }
    if (cityId !== "*") {
      if (commonOD !== "*") {
        bcData.push({ val: commonOD, title: "O&D" });
        self.setState({ selectedData: OD });
      }
    }
  };

  getMonthDrillDownData = (
    regionId,
    countryId,
    cityId,
    commonOD,
    loader,
    version
  ) => {
    var self = this;
    let {
      gettingMonth,
      getCabinValue,
      type,
      gettingYear,
      toggle,
      odsearchvalue,
    } = this.state;
    let selectedVersion = version ? version : "Null";
    if (loader === "circle_loader") {
      self.showLoader();
    } else {
      self.setState({
        loading: true,
        loading2: true,
        monthData: [],
        monthTotalData: [],
        drillDownData: [],
        drillDownTotalData: [],
      });
    }

    apiServices
      .getRPSMonthTables(
        regionId,
        countryId,
        cityId,
        commonOD,
        getCabinValue,
        selectedVersion
      )
      .then(function (result) {
        self.hideLoader();
        self.setState({ loading: false });
        if (result) {
          var totalData = result[0].totalData;
          var columnName = result[0].columnName;
          var columnNameTotal = result[0].columnNameTotal;
          var rowData = result[0].rowData;

          window.localStorage.setItem(
            "RPS Monthly Data",
            JSON.stringify(rowData)
          );
          originalMonthData = rowData;
          originalMonthTotalData = totalData;

          self.setState({
            monthData: self.getHighlightedMonth(
              rowData,
              gettingMonth,
              gettingYear
            ),
            monthcolumns: columnName,
            monthcolumnsTotal: columnNameTotal,
            monthTotalData: totalData,
            apiMonthlyData: result[0].apiMonthlyData,
            isActionPerformed: result[0].isActionPerformed,
            isAllApproved: result[0].isAllApproved,
          });
        }
      });

    if (loader === "circle_loader") {
      self.showLoader();
    }

    apiServices
      .getRPSDrillDown(
        gettingYear,
        gettingMonth,
        regionId,
        countryId,
        cityId,
        commonOD,
        getCabinValue,
        type,
        selectedVersion,
        odsearchvalue
      )
      .then((result) => {
        self.hideLoader();
        self.setState({ loading2: false });
        if (result) {
          originalDrilldownData = result[0].rowData;
          originalDrilldownTotalData = result[0].totalData;
          self.setState({
            drillDownTotalData: result[0].totalData,
            drillDownData: result[0].rowData,
            drillDownColumn: result[0].columnName,
            tabName:
              type === "Null" ? result[0].tabName : result[0].firstTabName,
            regionId: result[0].currentAccess.regionId,
            countryId: result[0].currentAccess.countryId,
            cityId: result[0].currentAccess.cityId,
            commonOD: result[0].currentAccess.commonOD,
            apiDrilldownData: result[0].apiDrilldownData,
          });
        }
      });
  };

  getDrillDownData = (regionId, countryId, cityId, commonOD, type) => {
    var self = this;
    let {
      gettingYear,
      gettingMonth,
      getCabinValue,
      isEdited,
      editedField,
      editedRowIndex,
      editedvalue,
      selectedVersion,
      odsearchvalue,
    } = this.state;

    apiServices
      .getRPSDrillDown(
        gettingYear,
        gettingMonth,
        regionId,
        countryId,
        cityId,
        commonOD,
        getCabinValue,
        type,
        selectedVersion,
        odsearchvalue
      )
      .then((result) => {
        self.setState({ loading2: false });
        if (result) {
          originalDrilldownData = result[0].rowData;
          originalDrilldownTotalData = result[0].totalData;
          self.setState(
            {
              drillDownTotalData: result[0].totalData,
              drillDownData: result[0].rowData,
              drillDownColumn: result[0].columnName,
              tabName:
                type === "Null" ? result[0].tabName : result[0].firstTabName,
              regionId: result[0].currentAccess.regionId,
              countryId: result[0].currentAccess.countryId,
              cityId: result[0].currentAccess.cityId,
              commonOD: result[0].currentAccess.commonOD,
              apiDrilldownData: result[0].apiDrilldownData,
            },
            () => {
              if (isEdited) {
                this.getEditedData(editedField, editedRowIndex, editedvalue);
              }
            }
          );
        }
      });
  };

  getMessages() {
    api
      .get(`usermessages?page_number=${this.state.count}`, "hideloader")
      .then((res) => {
        if (res.data.response) {
          const responseData = res.data.response;
          if (responseData.messages.length > 0) {
            this.setState({
              messageArray: this.state.messageArray.concat(
                responseData.messages
              ),
              msgResponseData: responseData,
            });
            apiServices
              .getUserPreferences(
                "rpsMessageCount",
                responseData.totalMessages,
                "count"
              )
              .then((result) => {
                this.setState({
                  msgCount: result,
                  msgCountVisible: result > 0,
                });
              });
          }
        }
      })
      .catch((err) => {
        console.log("rahul message err", err);
      });
  }

  fetchMoreData = () => {
    const { messageArray, msgResponseData, count } = this.state;
    if (messageArray.length >= msgResponseData.totalMessages) {
      this.setState({ hasMore: false });
      return;
    }
    let Count = count;
    Count = Count + 1;
    this.setState({ count: Count }, () => {
      setTimeout(() => {
        this.getMessages();
      }, 1500);
    });
  };

  getHighlightedMonth(monthData, month, year) {
    let monthNumber = window.monthNameToNum(month);
    let count = 0;
    let data = monthData.filter((data, index) => {
      data.highlightMe = false;
      var monthName = data.Month;
      const selectedMonth = `${window.shortMonthNumToName(
        monthNumber
      )} ${year}`;
      if (selectedMonth === monthName) {
        this.setState({ ensureIndexVisible: index - count });
        data.highlightMe = true;
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
    console.log(params.data.FRCT_R, "::::PARAMS:::::");
    var monththis = this;
    monththis.sendEvent("2", "clicked on Months row", "pos", "Pos Page");
    let {
      monthData,
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      type,
      isEdited,
      gettingMonth,
      selectedVersion,
    } = this.state;
    let selectedMonth = params.data.Month;
    let year = parseInt(params.data.Year);
    var column = params.colDef.field;
    this.setState({ rowDataforChannel: params.data.FRCT_R });
    if (
      isEdited &&
      (!(params.data.MonthName === gettingMonth) || column === "Month")
    ) {
      Swal.fire({
        title: "Submit before you proceed!",
        text: "Before switching to the next month or editing the other field you have to submit the previously edited data",
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "Submit",
        cancelButtonText: "Discard",
        cancelButtonColor: "#d33",
      }).then((result) => {
        if (result.value) {
          monththis.openMessageModal("Submit");
        } else if (result.dismiss === Swal.DismissReason.cancel) {
          params.api.updateRowData({
            update: this.getHighlightedMonth(
              originalMonthData,
              params.data.MonthName,
              year
            ),
          });

          monththis.setState({
            isEdited: false,
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
          window.localStorage.setItem(
            "rangeValueNextYear",
            JSON.stringify(range)
          );

          // monththis.setState({ isEdited: false, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
          monththis.getMonthDrillDownData(
            regionId,
            countryId,
            cityId,
            commonOD,
            "circle_loader",
            selectedVersion
          );
        }
      });
    } else {
      params.api.updateRowData({
        update: this.getHighlightedMonth(
          monthData,
          params.data.MonthName,
          year
        ),
      });
      //Getting Clubbed Data
      if (selectedMonth.includes(`Total ${currentYear - 1}`)) {
        this.setState(
          {
            showLastYearRows: !this.state.showLastYearRows,
            showNextYearRows: false,
          },
          () =>
            this.getLastYearClubbedData(
              this.state.showLastYearRows,
              selectedMonth
            )
        );
      } else if (selectedMonth.includes(`Total ${currentYear + 1}`)) {
        this.setState(
          {
            showNextYearRows: !this.state.showNextYearRows,
            showLastYearRows: false,
          },
          () =>
            this.getNextYearClubbedData(
              this.state.showNextYearRows,
              selectedMonth
            )
        );
      } else {
        monththis.setState({
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
        window.localStorage.setItem(
          "rangeValueNextYear",
          JSON.stringify(range)
        );
      }

      if (column === "Month" && !selectedMonth.includes("Total")) {
        monththis.setState({
          loading2: true,
          drillDownData: [],
          drillDownTotalData: [],
        });
        monththis.getDrillDownData(
          regionId,
          countryId,
          cityId,
          commonOD,
          type,
          selectedVersion
        );
      }
    }
  };

  regionCellClick = (params) => {
    var self = this;
    self.sendEvent("2", "clicked on Region drill down", "pos", "Pos Page");
    let {
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      selectedVersion,
    } = this.state;
    self.setState({ isEdited: false });
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
        this.storeValuesToLS(
          regionId,
          countryId,
          cityId,
          commonOD,
          getCabinValue,
          selectedDataWQ
        );

        if (!selectedTitle !== "O&D") {
          self.setState({ selectedData });
          bcData.push({ val: selectedDataWQ, title: selectedTitle });
        }

        if (regionId === "*") {
          self.getMonthDrillDownData(
            selectedData,
            countryId,
            cityId,
            commonOD,
            "",
            selectedVersion
          );
        } else if (countryId === "*") {
          self.getMonthDrillDownData(
            regionId,
            selectedData,
            cityId,
            commonOD,
            "",
            selectedVersion
          );
        } else if (cityId === "*") {
          self.getMonthDrillDownData(
            regionId,
            countryId,
            selectedData,
            commonOD,
            "",
            selectedVersion
          );
        } else if (commonOD === "*") {
          self.getMonthDrillDownData(
            regionId,
            countryId,
            cityId,
            selectedData,
            "",
            selectedVersion
          );
        }
      }
    }
  };

  rectifyURLValues(regionId, countryId, cityId) {
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
    if (Array.isArray(cityId)) {
      this.selectedCity = cityId.join(",");
    } else if (regionId.includes("','")) {
      this.selectedCity = cityId.split("','").join(",");
      this.selectedCity = this.selectedCity.substring(
        1,
        this.selectedCity.length - 1
      );
    } else {
      this.selectedCity = cityId;
      this.selectedCity = this.selectedCity.substring(
        1,
        this.selectedCity.length - 1
      );
    }
    // if (Array.isArray(commonOD)) {
    //   this.selectedOd = commonOD.join(",");
    // } else if (regionId.includes("','")) {
    //   this.selectedOd = commonOD.split("','").join(",");
    //   this.selectedOd = this.selectedOd.substring(
    //     1,
    //     this.selectedOd.length - 1
    //   );
    // } else {
    //   this.selectedOd = commonOD;
    //   this.selectedOd = this.selectedOd;
    // }
  }

  storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, data) {
    let region = [];
    let country = [];
    let city = [];
    // let od = [];
    let cabin = [];

    this.rectifyURLValues(regionId, countryId, cityId);

    if (regionId === "*") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(data)}`
      );
      region.push(data);
      window.localStorage.setItem("RegionSelected", JSON.stringify(region));
    } else if (countryId === "*") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${data}`
      );
      country.push(data);
      window.localStorage.setItem("CountrySelected", JSON.stringify(country));
    } else if (cityId === "*") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${this.selectedCountry}&POS=${data}`
      );
      city.push(data);
      window.localStorage.setItem("CitySelected", JSON.stringify(city));
    } else if (commonOD === "*") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${this.selectedCountry}&POS=${
          this.selectedCity
        }&${encodeURIComponent("O&D")}=${data}`
      );
      // od.push(data);
      window.localStorage.setItem("ODSelected", data);
    }
    // else if (getCabinValue === 'Null') {
    //   cabin.push(data)
    //   window.localStorage.setItem('CabinSelected', JSON.stringify(cabin))
    // }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    self.sendEvent("2", `clicked on ${selectedType} tab`, "pos", "Pos Page");
    let { regionId, countryId, cityId, commonOD, monthcolumns } = this.state;
    self.setState({
      type: selectedType,
      drillDownData: [],
      drillDownTotalData: [],
      loading2: true,
    });

    if (outerTab) {
      this.setState({ outerTab: true });
    } else {
      this.setState({ outerTab: false });
    }
    self.getDrillDownData(regionId, countryId, cityId, commonOD, selectedType);
    this.monthParams.setColumnDefs(monthcolumns);
  };

  homeHandleClick = (e) => {
    var self = this;
    self.sendEvent("2", "clicked on Network", "pos", "Pos Page");

    self.setState({
      loading: true,
      loading2: true,
      firstHome: false,
      posMonthDetails: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
      toggle: "bc",
    });

    window.localStorage.setItem("RegionSelected", "Null");
    window.localStorage.setItem("CountrySelected", "Null");
    window.localStorage.setItem("CitySelected", "Null");
    window.localStorage.setItem("ODSelected", "Null");

    self.getMonthDrillDownData(
      "*",
      "*",
      "*",
      "*",
      "",
      this.state.selectedVersion
    );

    bcData = [];
    var newURL = window.location.href.split("?")[0];
    window.history.pushState("object", document.title, newURL);
    // this.props.history.push('/pos')
  };

  listHandleClick = (data, title, selection) => {
    var self = this;
    self.sendEvent("2", "clicked on Drill down list", "pos", "Pos Page");
    let { regionId, countryId, cityId, selectedVersion } = this.state;
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
      posMonthDetails: [],
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
      this.changeURLOnListClick(regionId, countryId, cityId, data, getColName);
    } else if (selection === "browserBack") {
      this.onBackPressClearLS(getColName);
    }

    if (getColName === "Region") {
      self.getMonthDrillDownData(
        selectedData,
        "*",
        "*",
        "*",
        "",
        selectedVersion
      );
    } else if (getColName === "Country") {
      self.getMonthDrillDownData(
        regionId,
        selectedData,
        "*",
        "*",
        "",
        selectedVersion
      );
    } else if (getColName === "POS") {
      self.getMonthDrillDownData(
        regionId,
        countryId,
        selectedData,
        "*",
        "",
        selectedVersion
      );
    } else if (getColName === "O&D") {
      self.getMonthDrillDownData(
        regionId,
        countryId,
        cityId,
        selectedData,
        "",
        selectedVersion
      );
    }
  };

  changeURLOnListClick(regionId, countryId, cityId, selectedData, getColName) {
    this.rectifyURLValues(regionId, countryId, cityId);

    if (getColName === "Region") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(selectedData)}`
      );
      window.localStorage.setItem("CountrySelected", "Null");
      window.localStorage.setItem("CitySelected", "Null");
      window.localStorage.setItem("ODSelected", "Null");
    } else if (getColName === "Country") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${selectedData}`
      );
      window.localStorage.setItem("CitySelected", "Null");
      window.localStorage.setItem("ODSelected", "Null");
    } else if (getColName === "POS") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${this.selectedCountry}&POS=${selectedData}`
      );
      window.localStorage.setItem("ODSelected", "Null");
    } else if (getColName === "O&D") {
      this.props.history.push(
        `${this.pathName}?Region=${encodeURIComponent(
          this.selectedRegion
        )}&Country=${this.selectedCountry}&POS=${
          this.selectedCity
        }&${encodeURIComponent("O&D")}=${selectedData}`
      );
    }
  }

  onBackPressClearLS(getColName) {
    if (getColName === "Region") {
      window.localStorage.setItem("CountrySelected", "Null");
      window.localStorage.setItem("CitySelected", "Null");
      window.localStorage.setItem("ODSelected", "Null");
    } else if (getColName === "Country") {
      window.localStorage.setItem("CitySelected", "Null");
      window.localStorage.setItem("ODSelected", "Null");
    } else if (getColName === "POS") {
      window.localStorage.setItem("ODSelected", "Null");
    }
  }

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
    self.sendEvent("2", "clicked on Cabin drop down", "pos", "Pos Page");
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
      posMonthDetails: [],
      monthTotalData: [],
      drillDownData: [],
      drillDownTotalData: [],
    });
    let { regionId, countryId, cityId, commonOD, selectedVersion } = this.state;
    self.getMonthDrillDownData(
      regionId,
      countryId,
      cityId,
      commonOD,
      "",
      selectedVersion
    );
  }

  getSelectedRows = (params) => {
    const month = [];
    params.api
      .getSelectedRows()
      .filter((d) => month.push(window.monthNameToNum(d.MonthName)));
    this.setState({ selectedMonths: month });
  };

  onCellValueChanged = (params) => {
    const { gettingMonth, gettingYear } = this.state;
    const field = params.colDef.field;
    const rowIndex = params.rowIndex;
    const newValue = params.newValue.toString();
    const oldValue = params.oldValue.toString();
    originalMonthData = JSON.parse(
      window.localStorage.getItem("RPS Monthly Data")
    );
    if (newValue !== oldValue) {
      if (!Constant.validateNum(newValue) || !newValue) {
        Swal.fire({
          title: "Error!",
          text: "Enter absolute numbers only",
          icon: "error",
          confirmButtonText: "Ok",
        }).then(() => {
          this.setState({ monthData: originalMonthData });
          params.api.updateRowData({
            update: this.getHighlightedMonth(
              originalMonthData,
              gettingMonth,
              gettingYear
            ),
          });
        });
      } else {
        const convertedNewValue = parseFloat(newValue);
        this.setState({
          isEdited: true,
          isSubmitted: false,
          editedField: field,
          editedRowIndex: rowIndex,
          editedvalue: convertedNewValue,
        });
        this.getEditedData(field, rowIndex, convertedNewValue);
      }
    }
  };

  getEditedData(field, rowIndex, newValue) {
    const {
      apiMonthlyData,
      apiDrilldownData,
      gettingYear,
      gettingMonth,
      type,
    } = this.state;
    let editedRow = apiMonthlyData.TableData.filter((d, i) => i === rowIndex);
    let changedRow = {
      Month: editedRow[0].Month,
      Year: editedRow[0].Year,
      Passenger_EDIT:
        field === "Edit_P" ? newValue : editedRow[0].Passenger_EDIT,
      AverageFare_EDIT:
        field === "Edit_A" ? newValue : editedRow[0].AverageFare_EDIT,
      Revenue_CY: editedRow[0].Revenue_CY,
    };
    console.log("::changedRow::", changedRow);

    let data = {
      changedRow: changedRow,
      monthlyData: {
        TableData: apiMonthlyData.TableData,
        Total: apiMonthlyData.Total,
      },
      drilldownData: {
        TableData: apiDrilldownData.TableData,
        Total: apiDrilldownData.Total,
        ColumName: apiDrilldownData.ColumName,
        first_ColumName: apiDrilldownData.first_ColumName,
        CurrentAccess: {
          regionId: apiDrilldownData.CurrentAccess.regionId,
          countryId: apiDrilldownData.CurrentAccess.countryId,
          cityId: apiDrilldownData.CurrentAccess.cityId,
          commonOD: apiDrilldownData.CurrentAccess.commonOD,
          base_access: apiDrilldownData.CurrentAccess.base_access,
        },
      },
    };

    api
      .post(`rpsdisplaycalculations`, data)
      .then((res) => {
        if (res.data.response) {
          const responseData = res.data.response;
          const monthlyData = responseData.monthlyData;
          const drillDownData = responseData.drilldownData;
          const monthdata = apiServices.getRPSData(monthlyData.TableData);
          this.setState({
            monthData: this.getHighlightedMonth(
              monthdata,
              gettingMonth,
              gettingYear
            ),
            monthTotalData: apiServices.getRPSData(monthlyData.Total, "total"),
            drillDownData: apiServices.getRPSData(drillDownData.TableData),
            drillDownTotalData: apiServices.getRPSData(
              drillDownData.Total,
              "total"
            ),
            apiMonthlyData: monthlyData,
            apiDrilldownData: drillDownData,
          });
          this.getFlashingCells(field, rowIndex);
        }
      })
      .catch((err) => {
        console.log("rahul", err);
        Swal.fire({
          title: "Error",
          text: `Something went wrong`,
          icon: "error",
          confirmButtonText: "Ok",
        });
      });
  }

  getFlashingCells(field, rowIndex) {
    const rowNodeM = this.monthParams.getDisplayedRowAtIndex(rowIndex);
    this.monthParams.flashCells({
      rowNodes: [rowNodeM],
      columns: [field, "FRCT_R"],
    });
    const rowNodeT = this.monthTotalParams.getDisplayedRowAtIndex(0);
    this.monthTotalParams.flashCells({
      rowNodes: [rowNodeT],
      columns: [field, "FRCT_R"],
    });
    this.drillDownParams.flashCells({
      columns: [field, "FRCT_R"],
    });
    const rowNodeDT = this.drillDownTotalParams.getDisplayedRowAtIndex(0);
    this.drillDownTotalParams.flashCells({
      rowNodes: [rowNodeDT],
      columns: [field, "FRCT_R"],
    });
  }

  getMonthParams = (params) => {
    this.monthParams = params;
  };

  getMonthTotalParams = (params) => {
    this.monthTotalParams = params;
  };

  getDrillDownParams = (params) => {
    this.drillDownParams = params;
  };

  getDrillDownTotalParams = (params) => {
    this.drillDownTotalParams = params;
  };

  odsearchfilter = (e) => {
    const { regionId, countryId, cityId, commonOD, type, odsearchvalue } =
      this.state;
    var ODvalue = e.target.value;
    this.setState({ odsearchvalue: ODvalue });
    // var grid =  e.api.setQuickFilter(document.getElementById('filter-text-box').value);
    console.log(odsearchvalue, "option");
    this.setState({ loading2: true });
    setTimeout(() => {
      this.getDrillDownData(regionId, countryId, cityId, commonOD, type);
    }, 4000);
  };

  // odsearchfilter = (e) => {
  //   gridApi.setQuickFilter(e.target.value)
  // }

  showLoader = () => {
    $("#loaderImage").addClass("loader-visible");
  };

  hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible");
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
  };

  renderTabs = () => {
    let {
      gettingMonth,
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      tabName,
      gettingYear,
      type,
      outerTab,
      ancillary,
    } = this.state;
    // const downloadURLDrillDown = apiServices.exportCSVPOSDrillDownURL(gettingYear, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type)
    // const downloadURLMonthly = apiServices.exportCSVPOSMonthlyURL(regionId, countryId, cityId, commonOD, getCabinValue)

    return (
      <ul className="nav nav-tabs" role="tablist">
        {tabName === "Cabin" ? (
          <li
            role="presentation"
            onClick={() => this.tabClick("OD", "outerTab")}
          >
            <a
              href="#Section2"
              aria-controls="profile"
              role="tab"
              data-toggle="tab"
            >
              O&D
            </a>
          </li>
        ) : (
          ""
        )}

        <li
          role="presentation"
          className={`${ancillary ? "" : "active"}`}
          onClick={() => this.tabClick("Null")}
        >
          <a href="#Section1" aria-controls="home" role="tab" data-toggle="tab">
            {tabName}
          </a>
        </li>

        {tabName === "O&D" ? (
          ""
        ) : tabName === "Cabin" ? (
          ""
        ) : (
          <li
            role="presentation"
            onClick={() => this.tabClick("OD")}
            className={outerTab ? "active" : ""}
          >
            <a
              href="#Section2"
              aria-controls="profile"
              role="tab"
              data-toggle="tab"
            >
              O&D
            </a>
          </li>
        )}

        {tabName === "Cabin" ? (
          ""
        ) : (
          <li role="presentation" onClick={() => this.tabClick("Cabin")}>
            <a
              href="#Section3"
              aria-controls="messages"
              role="tab"
              data-toggle="tab"
            >
              Cabin
            </a>
          </li>
        )}

        {/* <li role="presentation" onClick={() => this.tabClick('Segment')}>
          <a href="#Section4" aria-controls="messages" role="tab" data-toggle="tab">
            Segment
          </a></li> */}

        <li role="presentation" onClick={() => this.tabClick("Channel")}>
          <a
            href="#Section5"
            aria-controls="messages"
            role="tab"
            data-toggle="tab"
          >
            Channel
          </a>
        </li>

        {/* <DownloadCSV url={downloadURLDrillDown} name={`POS DRILLDOWN`} path={`/pos`} page={`Pos Page`} />
        <DownloadCSV url={downloadURLMonthly} name={`POS MONTHLY`} path={`/pos`} page={`Pos Page`} /> */}
      </ul>
    );
  };

  openMessageModal = (action) => {
    this.setState({ displayModal: true, action: action });
  };

  LoadFactorModel = () => {
    this.setState({ displayModal2: true });
  };

  RouteConversion = () => {
    Swal.fire({
      title: "Processing",
      text: `Route Conversion Process has Started. You can Continue We will notify when its done`,
      icon: "info",
      confirmButtonText: "Ok",
    });
    Swal.showLoading();
    this.setState({ disableRouteConversion: true });
    axios
      .get(
        `${API_URL}/DemandRouteConversion?flownValue=${this.flownValue}&flownValueLength=${this.flownValueLength}&CurrentYear=${this.Cyear}&TargetYear=${this.Tyear}&Version=${this.state.selectedVersion}`,
        this.getDefaultHeader()
      )
      .then((response) => {
        Swal.fire({
          title: "Completed",
          text: `Process has been Completed`,
          icon: "success",
          confirmButtonText: "Ok",
        });
        this.props.history.push(
          `/rpsRoute${Constant.getRouteFiltersSearchURL()}`
        );
        console.log(response, "correct");
      })
      .catch((error) => {
        Swal.fire({
          title: "Failed",
          text: `Process has been failed Please Retry else Contact Revemax`,
          icon: "error",
          confirmButtonText: "Ok",
        });
        this.setState({ disableRouteConversion: false });
        console.log(error, "wrong");
      });
  };

  OptimizedRouteConversion = () => {
    this.setState({ displayModal2: false });
    Swal.fire({
      title: "Processing",
      text: `Optimized Route Conversion Process has Started. You can Continue We will notify when its done`,
      icon: "info",
      confirmButtonText: "Ok",
    });
    Swal.showLoading();
    this.setState({ disableOptimzedRouteConversion: true });
    axios
      .get(
        `${API_URL}/OptimisedRouteConversion?flownValue=${this.flownValue}&flownValueLength=${this.flownValueLength}&CurrentYear=${this.Cyear}&TargetYear=${this.Tyear}&Loadfactor=${this.state.LoadFactorValue}&Version=${this.state.selectedVersion}`,
        this.getDefaultHeader()
      )
      .then((response) => {
        Swal.fire({
          title: "Completed",
          text: `Process has been Completed`,
          icon: "success",
          confirmButtonText: "Ok",
        });
        this.props.history.push(
          `/rpsRoute${Constant.getRouteFiltersSearchURL()}`
        );
        console.log(response, "correct");
      })
      .catch((error) => {
        Swal.fire({
          title: "Failed",
          text: `Process has been failed Please Retry else Contact Revemax`,
          icon: "error",
          confirmButtonText: "Ok",
        });
        this.setState({ disableOptimzedRouteConversion: false });
        console.log(error, "wrong");
      });
  };

  DemandOptimization = () => {
    Swal.fire({
      title: "Processing",
      text: `Network optimization process has been started, once the process complete we will notify you.`,
      icon: "info",
      confirmButtonText: "Ok",
    });
    this.setState({ disableOptimization: true });
    Swal.showLoading();
    axios
      .get(
        `${API_URL}/DemmandOptimization?flownValue=${this.flownValue}&flownValueLength=${this.flownValueLength}&CurrentYear=${this.Cyear}&TargetYear=${this.Tyear}&channelYear=${this.Channelyear}&segmentationYear=${this.Segmentationyear}&version='Null'`,
        this.getDefaultHeader()
      )
      .then((response) => {
        Swal.fire({
          title: "Completed",
          text: `Process has been Completed`,
          icon: "success",
          confirmButtonText: "Ok",
        });
        this.setState({ btndisable: true });
        console.log(response, "correct");
        setTimeout(() => {
          window.location.reload();
        }, 100);
      })
      .catch((error) => {
        Swal.fire({
          title: "Failed",
          text: `Process has been failed Please Retry else Contact Revemax`,
          icon: "error",
          confirmButtonText: "Ok",
        });
        this.setState({ disableRouteConversion: false });
        console.log(error, "wrong");
      });
  };

  _validateMsg = (event) => {
    this.setState({ disable: false });
    let message = this.refs.message.value.trim();
    if (message === "") {
      this.setState({
        errorMsg: "Please enter message",
      });
    } else {
      this.setState({
        errorMsg: "",
        disable: true,
      });
    }
  };

  updateData() {
    const {
      apiMonthlyData,
      apiDrilldownData,
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      type,
      editedRowIndex,
      action,
      selectedNewVersion,
      selectedVersion,
    } = this.state;
    let editedRow = apiMonthlyData.TableData.filter(
      (d, i) => i === editedRowIndex
    );
    let version =
      selectedNewVersion === "Null" ? selectedVersion : selectedNewVersion;
    let data = {
      Month: editedRow[0].Month,
      Year: editedRow[0].Year,
      Passenger_EDIT: editedRow[0].Passenger_EDIT,
      AverageFare_EDIT: editedRow[0].AverageFare_EDIT,
      Revenue_CY: editedRow[0].Revenue_CY,
      action_performed: action,
      action_level: this.userData.accessLevelPOS,
      version: version,
    };

    api
      .post(
        `rpsupdatetable?${apiServices.Params(
          regionId,
          countryId,
          cityId,
          getCabinValue
        )}&commonOD=${commonOD}`,
        data
      )
      .then((res) => {
        if (res.data.response) {
          const responseData = res.data.response;
          this.setState({ isEdited: false, isSubmitted: true });
          Swal.fire({
            title: "Submitted",
            text: `Data is submitted successfully`,
            icon: "success",
            confirmButtonText: "Ok",
          }).then(() => {
            this.setState({ messageArray: [] }, () => this.getMessages());
            this.getMonthDrillDownData(
              regionId,
              countryId,
              cityId,
              commonOD,
              "circle_loader",
              version
            );
          });
        }
      })
      .catch((err) => {
        console.log("rahul", err);
        Swal.fire({
          title: "Error",
          text: `Something went wrong`,
          icon: "error",
          confirmButtonText: "Ok",
        });
      });
  }

  postMessage() {
    const {
      regionId,
      countryId,
      cityId,
      commonOD,
      getCabinValue,
      type,
      editedRowIndex,
      action,
      selectedMonths,
      selectedVersion,
    } = this.state;
    const months = action === "Submit" ? [editedRowIndex + 1] : selectedMonths;
    let data = {
      userName: this.userData.userDetails.username,
      message: this.refs.message.value.trim(),
      month: months,
      action: action,
      action_level: this.userData.accessLevelPOS,
      access: this.userData.userDetails.access,
    };
    this.resetfields();
    api
      .post(
        `usermessages?${apiServices.Params(
          regionId,
          countryId,
          cityId,
          getCabinValue
        )}&commonOD=${commonOD}`,
        data
      )
      .then((res) => {
        if (res.data.response) {
          const responseData = res.data.response;
          if (action === "Submit") {
            this.updateData();
          } else {
            Swal.fire({
              title: "Submitted",
              text: `Message is Submitted successfully`,
              icon: "success",
              confirmButtonText: "Ok",
            }).then(() => {
              this.getMonthDrillDownData(
                regionId,
                countryId,
                cityId,
                commonOD,
                "circle_loader",
                selectedVersion
              );
              this.setState({ messageArray: [] }, () => this.getMessages());
            });
          }
        }
      })
      .catch((err) => {
        console.log("rahul", err);
        Swal.fire({
          title: "Error",
          text: `Something went wrong`,
          icon: "error",
          confirmButtonText: "Ok",
        });
      });
  }

  actionButton() {
    this.setState({ displayModal: false, errorMsg: "", disable: false });
    this.postMessage();
  }

  resetfields() {
    this.setState({ displayModal: false, errorMsg: "", disable: false });
  }

  handleChange = (event, e) => {
    this.setState({ LoadFactorValue: e });
  };

  renderMessageModal = () => {
    return (
      <Modal
        show={this.state.displayModal}
        aria-labelledby="ModalHeader"
        onHide={() => this.setState({ displayModal: false })}
      >
        <Modal.Header>
          <Modal.Title id="ModalHeader">{`${this.state.action} Message`}</Modal.Title>
        </Modal.Header>
        <Modal.Body className="resetpass-body">
          <label htmlFor="message">{`Drop your message here* :`}</label>
          <textarea
            type="message"
            ref="message"
            className="form-control"
            onChange={() => this._validateMsg()}
          />
          <p>{this.state.errorMsg}</p>
        </Modal.Body>
        <Modal.Footer>
          <button
            type="button"
            className="btn btn-success"
            onClick={() => this.actionButton()}
            disabled={!this.state.disable}
          >
            {this.state.action}
          </button>
          <button
            type="button"
            className="btn btn-danger"
            onClick={() => this.resetfields()}
          >
            Cancel
          </button>
        </Modal.Footer>
      </Modal>
    );
  };

  renderSelectionModal = () => {
    return (
      <Modal
        show={this.state.displayModal2}
        aria-labelledby="ModalHeader"
        onHide={() => this.setState({ displayModal2: false })}
      >
        <Modal.Header>
          <Modal.Title id="ModalHeader"> Load Factor </Modal.Title>
        </Modal.Header>
        <Modal.Body className="resetpass-body">
          <label htmlFor="message">{`Please Select Load Factor Value :`}</label>
          <div className="Boxes1">
            <div style={{ marginLeft: "1%" }}>
              <Typography style={{ marginBottom: "6%", fontSize: "2vh" }}>
                Load Factor
              </Typography>
              <Box
                sx={{
                  width: "100%",
                  display: "flex",
                  marginLeft: "4%",
                  "& .MuiSlider-thumb": {
                    height: "12px",
                    width: "10px",
                  },
                  "& .MuiSlider-track": {
                    height: 3,
                  },
                }}
              >
                <Slider
                  getAriaLabel={() => "Temperature range"}
                  min={0}
                  max={100}
                  value={this.state.LoadFactorValue}
                  onChange={(event, e) => this.handleChange(event, e)}
                  valueLabelDisplay="on"
                  getAriaValueText={this.valuetext}
                />
              </Box>
            </div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <button
            type="button"
            className="btn btn-success"
            onClick={() => this.OptimizedRouteConversion()}
          >
            Proceed
          </button>
          {/* <button type="button" className="btn btn-danger" onClick={() => this.resetfields()} >Cancel</button> */}
        </Modal.Footer>
      </Modal>
    );
  };

  redirection = (e) => {
    this.sendEvent(
      "2",
      "clicked on POS/Route RPS drop down",
      "/rpsPOS",
      "RPS POS Page"
    );
    let name = e.target.value;
    console.log("rahul", name);
    if (name === "Route") {
      this.props.history.push(
        `/rpsRoute${Constant.getRouteFiltersSearchURL()}`
      );
      bcData = [];
    } else {
      this.props.history.push("/rpsPos");
      bcData = [];
    }
  };

  versionHandling = (e) => {
    const { regionId, countryId, cityId, commonOD } = this.state;
    let selectedVersion = e.target.value;
    this.setState({ selectedVersion });
    this.getMonthDrillDownData(
      regionId,
      countryId,
      cityId,
      commonOD,
      "circle_loader",
      selectedVersion
    );
    window.localStorage.setItem("RPSVersion", selectedVersion);
  };

  handleDownload = () => {
    const {
      selectedVersion,
      regionId,
      countryId,
      cityId,
      commonOD,
      selectedNewVersion,
      rowDataforChannel,
    } = this.state;

    // console.log(originalDrilldownTotalData[0].FRCT_R, ":::::HELLO::::::")

    if (rowDataforChannel !== originalDrilldownTotalData[0].FRCT_R) {
      Swal.fire({
        title: "Warning",
        text: `Channel Percentage Value must be equal to 100 !`,
        icon: "warning",
        confirmButtonText: "Ok",
      });
    } else {
      let data = {
        version: selectedVersion,
      };

      api
        .post(`UpdatedchannelTable`, data)
        .then((res) => {
          console.log(res);
          Swal.fire({
            title: "Success",
            text: `This Version Fetched Successfully`,
            icon: "success",
            confirmButtonText: "Ok",
          });
        })
        .catch((err) => {
          console.log(err);
          Swal.fire({
            title: "Error",
            text: `Something went wrong. Please try again`,
            icon: "error",
            confirmButtonText: "Ok",
          });
        });
    }
  };
  postVersionAction(action) {
    const {
      selectedVersion,
      regionId,
      countryId,
      cityId,
      commonOD,
      selectedNewVersion,
    } = this.state;
    let newVersion =
      selectedNewVersion === "Null" ? selectedVersion : selectedNewVersion;
    let data = {
      version: action === "Freeze" ? newVersion : selectedVersion,
      action: action,
    };

    api
      .post(`freezeandunfreeze`, data)
      .then((res) => {
        if (res.data.response) {
          console.log("::HELLO::", res.data.response, version);
          const version = res.data.response;
          if (version) {
            Swal.fire({
              title: "Success",
              text: `This Version ${action} successfully`,
              icon: "success",
              confirmButtonText: "Ok",
            }).then(() => {
              this.setState({ version_action: action, isSubmitted: false });
              if (action === "Freeze") {
                this.getVersionList("afterFreeze");
                this.getMonthDrillDownData(
                  regionId,
                  countryId,
                  cityId,
                  commonOD,
                  "circle_loader",
                  newVersion
                );
                window.localStorage.setItem("RPSVersion", newVersion);
              } else {
                this.setState({
                  selectedNewVersion: version,
                });
                this.getMonthDrillDownData(
                  regionId,
                  countryId,
                  cityId,
                  commonOD,
                  "circle_loader",
                  version
                );
                window.localStorage.setItem("RPSVersion", version);
              }
            });
          }
        }
      })
      .catch((err) => {
        Swal.fire({
          title: "Error",
          text: `Something went wrong. Please try again`,
          icon: "error",
          confirmButtonText: "Ok",
        });
      });
  }

  render() {
    const downloadURL = localStorage.getItem("RPSPOSDownloadURL");
    const {
      cabinOption,
      cabinSelectedDropDown,
      cabinDisable,
      accessLevelDisable,
      regionId,
      countryId,
      cityId,
      commonOD,
      isEdited,
      isActionPerformed,
      selectedVersion,
      version_action,
      isSubmitted,
      versionList,
      type,
      rpsValues,
      disableOptimization,
      disableOptimzedRouteConversion,
      disableRouteConversion,
    } = this.state;
    const { userData } = this;
    return (
      <div className="rps">
        <TopMenuBar dashboardPath={`/posDashboard`} {...this.props} />
        {/* <TopMenuBar dashboardPath={'/posRevenuePlanning'} {...this.props} /> */}
        <Loader />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: "0px" }}>
              <div className="col-md-4 col-sm-4 col-xs-4 toggle1">
                <select
                  className="form-control cabinselect pos-route-dropdown"
                  onChange={(e) => this.redirection(e)}
                >
                  <option value="POS" selected={true}>
                    POS RPS
                  </option>
                  <option value="Route">Route RPS</option>
                </select>
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      <div
                        style={{
                          cursor: accessLevelDisable
                            ? "not-allowed"
                            : "pointer",
                        }}
                      >
                        <li
                          className={`${
                            accessLevelDisable ? "breadcrumb-disable" : ""
                          }`}
                          onClick={() => this.homeHandleClick()}
                        >
                          {" "}
                          {"Network"}{" "}
                        </li>
                      </div>
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

                      {this.state.versionList.length == 0 &&
                      regionId === "*" &&
                      countryId === "*" &&
                      cityId === "*" &&
                      commonOD === "*" ? (
                        <div style={{ width: "max-content" }}>
                          <h2 style={{ marginLeft: "20px" }}>
                            Demand Estimation Screen
                          </h2>
                        </div>
                      ) : (
                        ""
                      )}

                      {regionId === "*" &&
                      countryId === "*" &&
                      cityId === "*" &&
                      commonOD === "*" &&
                      this.state.versionList.length != 0 ? (
                        <div style={{ width: "max-content" }}>
                          <h2 style={{ marginLeft: "20px" }}>
                            Network Optimal Target Screen
                          </h2>
                        </div>
                      ) : (
                        ""
                      )}
                    </ol>
                  </nav>
                </section>
              </div>
              <div
                className="col-md-5 col-sm-5 col-xs-12 toggle2"
                style={{ marginRight: "20px" }}
              >
                {userData.isPOSNetworkAdmin && versionList.length > 0 ? (
                  <div className="cabin-selection">
                    <h4>Version :</h4>
                    <select
                      className="form-control cabinselect currency-dropdown"
                      onChange={(e) => this.versionHandling(e)}
                      // disabled={version_action === "Unfreeze"}
                    >
                      {versionList.map((d) => (
                        <option
                          value={d.Version}
                          selected={selectedVersion === d.Version}
                        >
                          {d.Version}
                        </option>
                      ))}
                    </select>
                  </div>
                ) : (
                  ""
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
              </div>
            </div>
          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div className="x_panel">
              <div className="x_content">
                <DataTableComponent
                  rowData={this.state.monthData}
                  columnDefs={this.state.monthcolumns}
                  loading={this.state.loading}
                  suppressRowClickSelection={true}
                  onSelectionChanged={(params) => this.getSelectedRows(params)}
                  onCellValueChanged={(params) =>
                    this.onCellValueChanged(params)
                  }
                  rowSelection="multiple"
                  // frameworkComponents={{
                  //   customHeaderGroupComponent: RPSPosDownloadCustomHeaderGroup,
                  // }}
                  rowClassRules={this.state.monthRowClassRule}
                  onCellClicked={(cellData) =>
                    this.monthWiseCellClick(cellData)
                  }
                  ensureIndexVisible={this.state.ensureIndexVisible}
                  gridApi={this.getMonthParams}
                />
                <TotalRow
                  suppressRowClickSelection={true}
                  rowData={this.state.monthTotalData}
                  columnDefs={this.state.monthcolumnsTotal}
                  // frameworkComponents={{
                  //   customHeaderGroupComponent: RPSPosDownloadCustomHeaderGroup,
                  // }}
                  loading={this.state.loading}
                  gridApi={this.getMonthTotalParams}
                />
              </div>
              <div className="x_content2">
                <div
                  className="tab"
                  id="posTableTab"
                  role="tabpanel"
                  style={{ marginTop: "10px" }}
                >
                  {this.renderTabs()}

                  <div className="tab-content tabs">
                    {/* Region */}
                    <div
                      role="tabpanel"
                      className={`tab-pane fade in active`}
                      id="Section1"
                    >
                      {this.state.cityId != "*" ? (
                        <div
                          style={{ height: "4vh", backgroundColor: "#1784c7" }}
                        >
                          <input
                            placeholder="Search OD"
                            onChange={this.odsearchfilter}
                            id="filter-text-box"
                            style={{ width: "6vw", height: "4vh" }}
                            value={this.state.odsearchvalue}
                            // disabled={this.state.cityId != '*' ? false : true}
                          ></input>
                        </div>
                      ) : (
                        ""
                      )}
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        suppressRowClickSelection={true}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) =>
                          this.regionCellClick(cellData)
                        }
                        loading={this.state.loading2}
                        gridApi={this.getDrillDownParams}
                      />
                      <TotalRow
                        suppressRowClickSelection={true}
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                        gridApi={this.getDrillDownTotalParams}
                      />
                    </div>

                    {/* OD */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section2"
                    >
                      <div
                        style={{ height: "4vh", backgroundColor: "#1784c7" }}
                      >
                        <input
                          placeholder="Search OD"
                          onChange={this.odsearchfilter}
                          id="filter-text-box"
                          style={{ width: "6vw", height: "4vh" }}
                          value={this.state.odsearchvalue}
                          // disabled={this.state.cityId != '*' ? false : true}
                        ></input>
                      </div>
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        suppressRowClickSelection={true}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        pos={true}
                        gridApi={this.getDrillDownParams}
                      />
                      <TotalRow
                        suppressRowClickSelection={true}
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                        gridApi={this.getDrillDownTotalParams}
                      />
                    </div>

                    {/* Cabin */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section3"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        suppressRowClickSelection={true}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        pos={true}
                        gridApi={this.getDrillDownParams}
                      />
                      <TotalRow
                        suppressRowClickSelection={true}
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                        gridApi={this.getDrillDownTotalParams}
                      />
                    </div>

                    {/* Segment */}
                    {/* <div role="tabpanel" className="tab-pane fade" id="Section4">
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        suppressRowClickSelection={true}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        pos={true}
                        gridApi={this.getDrillDownParams}
                      />
                      <TotalRow
                        suppressRowClickSelection={true}
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                        gridApi={this.getDrillDownTotalParams}
                      />
                    </div> */}

                    {/* Channel */}
                    <div
                      role="tabpanel"
                      className="tab-pane fade"
                      id="Section5"
                    >
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        suppressRowClickSelection={true}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        pos={true}
                        gridApi={this.getDrillDownParams}
                      />
                      <TotalRow
                        suppressRowClickSelection={true}
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                        gridApi={this.getDrillDownTotalParams}
                      />
                    </div>
                  </div>
                </div>
              </div>
              {this.state.loading2 ? (
                ""
              ) : (
                <div className="btn-main">
                  {regionId === "*" && this.state.versionList.length == 0 ? (
                    <button
                      type="button"
                      className="btn search"
                      disabled={disableRouteConversion}
                      onClick={() => this.RouteConversion()}
                    >
                      {" "}
                      Route Conversion{" "}
                    </button>
                  ) : (
                    ""
                  )}
                  {regionId === "*" && this.state.versionList.length != 0 ? (
                    <button
                      type="button"
                      style={{ width: "200px" }}
                      className="btn search"
                      disabled={disableOptimzedRouteConversion}
                      onClick={() => this.LoadFactorModel()}
                    >
                      {" "}
                      Optimized Route Conversion{" "}
                    </button>
                  ) : (
                    ""
                  )}
                  {this.state.versionList.length == 0 && regionId === "*" ? (
                    <button
                      type="button"
                      className="btn search"
                      disabled={disableOptimization}
                      onClick={() => this.DemandOptimization()}
                    >
                      {" "}
                      Optimization{" "}
                    </button>
                  ) : (
                    ""
                  )}

                  {userData.isPOSNetworkAdmin &&
                  versionList.length > 0 &&
                  regionId === "*" ? (
                    version_action === "Freeze" ? (
                      <button
                        type="button"
                        className="btn search"
                        onClick={() => this.postVersionAction("Unfreeze")}
                      >
                        {" "}
                        Unfreeze
                      </button>
                    ) : (
                      <div>
                        <button
                          type="button"
                          className="btn search"
                          disabled={!isEdited}
                          onClick={() => this.openMessageModal("Submit")}
                        >
                          {" "}
                          Submit
                        </button>
                        <button
                          type="button"
                          className="btn search"
                          disabled={!isSubmitted}
                          onClick={() => this.postVersionAction("Freeze")}
                        >
                          {" "}
                          Freeze
                        </button>
                      </div>
                    )
                  ) : (
                    <div>
                      {!isEdited ? (
                        <div>
                          {userData.canApproveRPS && isActionPerformed ? (
                            <button
                              type="button"
                              className="btn search"
                              onClick={() => this.openMessageModal("Approve")}
                            >
                              {" "}
                              Approve
                            </button>
                          ) : (
                            ""
                          )}
                          {userData.canRejectRPS && isActionPerformed ? (
                            <button
                              type="button"
                              className="btn search"
                              onClick={() => this.openMessageModal("Reject")}
                            >
                              {" "}
                              Reject
                            </button>
                          ) : (
                            ""
                          )}
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="btn search"
                          onClick={() => this.openMessageModal("Submit")}
                        >
                          {" "}
                          Submit
                        </button>
                      )}
                    </div>
                  )}

                  {type === "Channel" && this.state.versionList.length != 0 ? (
                    <button
                      type="button"
                      style={{ width: "200px" }}
                      className="btn search"
                      disabled={disableOptimzedRouteConversion}
                      onClick={() => this.handleDownload()}
                    >
                      {" "}
                      Channel Target Update{" "}
                    </button>
                  ) : (
                    ""
                  )}
                  <NotificationRightSidebar
                    messageArray={this.state.messageArray}
                    msgResponseData={this.state.msgResponseData}
                    msgCount={this.state.msgCount}
                    msgCountVisible={this.state.msgCountVisible}
                    hasMore={this.state.hasMore}
                    fetchMoreData={this.fetchMoreData}
                    {...this.props}
                  />
                </div>
              )}
            </div>
          </div>
        </div>

        <div>
          {this.renderMessageModal()}

          {this.renderSelectionModal()}

          <DatatableModelDetails
            tableModalVisible={this.state.tableModalVisible}
            datas={this.state.modalData}
            rowData={this.state.modalCompartmentData}
            columns={this.state.modalCompartmentColumn}
            header={`${this.state.gettingMonth}`}
            loading={this.state.loading3}
            closeTableModal={() => this.closeTableModal()}
          />
          <ChartModelDetails
            chartVisible={this.state.chartVisible}
            datas={this.state.modeldrillDownDatas}
            columns={this.state.modeldrillDownColumn}
            closeChartModal={() => this.closeChartModal()}
          />
        </div>
      </div>
    );
  }
}
const NewComponentRPS = BrowserToProps(RPSPos);

export default NewComponentRPS;
