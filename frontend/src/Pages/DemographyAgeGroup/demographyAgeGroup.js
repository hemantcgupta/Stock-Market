import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import api from '../../API/api'
import AlertModal from '../../Component/AlertModal';
import InfareMultiLineChart from '../../Component/InfareMultiLineChart';
import DatatableModelDetails from '../../Component/dataTableModel';
import DataTableModelDemoSegment from '../../Component/dataTableModelDemoSegment';
import ChartModelDetails from '../../Component/chartModel';
import Loader from '../../Component/Loader';
import cookieStorage from '../../Constants/cookie-storage';
import String from '../../Constants/validator';
import URLSearchParams from '../../Constants/validator';
import Access from '../../Constants/accessValidation'
import DownloadCSV from '../../Component/DownloadCSV';
import DataTableComponent from '../../Component/DataTableComponent';
import TotalRow from '../../Component/TotalRow';
import { string } from '../../Constants/string';
import color from '../../Constants/color'
import Constant from '../../Constants/validator';
import $ from 'jquery';
import '../../App';
import './demographyAgeGroup.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import POSCustomHeaderGroup from './POSCustomHeaderGroup';
import Modal from 'react-bootstrap-modal';
import Input from "@material-ui/core/Input";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import ListItemText from "@material-ui/core/ListItemText";
import Select from "@material-ui/core/Select";
import Checkbox from "@material-ui/core/Checkbox";
import BrowserToProps from 'react-browser-to-props';
import Swal from 'sweetalert2';
import DemographyGraphsPage from './DemographyGraphs'


const apiServices = new APIServices();
const MonthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const currentYear = new Date().getFullYear()
const CurrentMonth = MonthNames[new Date().getMonth()] + ' ' + currentYear;

let monthData = [];
let bcData = [];

class POSDetail extends Component {
  constructor(props) {
    super(props);
    this.pathName = window.location.pathname;
    this.selectedRegion = null;
    this.selectedCountry = null;
    this.selectedCity = null;
    this.selectedOD = null;
    this.selectedCabin = null;
    this.selectedCustomerSegment = null;
    this.gridApiMonth = null;
    this.state = {
      posMonthDetails: [],
      monthTotalData: [],
      drillDownTotalData: [],
      monthcolumns: [],
      drillDownColumn: [],
      drillDownData: [],
      modaldrillDownColumn: [],
      modaldrillDownData: [],
      modalData: [],
      availModalData: [],
      availModalColumn: [],
      availTotalData: [],
      availModalVisible: false,
      clicked: false,
      tableDatas: true,
      gettingMonth: null,
      gettingYear: null,
      gettingRegion: 'Null',
      monthTableTitle: 'NETWORK',
      tabLevel: 'Null',
      cabinOption: [],
      alertData: [],
      getCabinValue: [],
      cabinSelectedDropDown: [],
      cabinDisable: true,
      toggle: 'bc',
      alertVisible: false,
      tableModalVisible: false,
      tabName: 'Region',
      regionId: '*',
      countryId: '*',
      cityId: '*',
      commonOD: '*',
      cabinId: '*',
      enrichId: '*',
      customerSegmentationId: '*',
      customerSegmentationId_1: '*',
      type: 'Null',
      NationalityId: '*',
      AgeBandId: '*',
      baseAccess: '',
      selectedData: 'Null',
      infareData: [],
      infareModalVisible: false,
      infareGraphHeading: '',
      loading: false,
      loading2: false,
      loading3: false,
      firstLoadList: false,
      showLastYearRows: false,
      showNextYearRows: false,
      accessLevelDisable: false,
      chartVisible: false,
      forecastChartHeader: '',
      posMonthRowClassRule: {
        'highlight-row': 'data.highlightMe',
      },
      firstHome: true,
      infareCurrency: '',
      outerTab: false,
      ancillary: false,
      ensureIndexVisible: null
    }
    this.sendEvent('1', 'viewed Pos Page', 'pos', 'Pos page');
  }

  sendEvent = (id, description, path, page_name) => {
    var eventData = {
      event_id: `${id}`,
      description: `User ${description}`,
      where_path: `/${path}`,
      page_name: `${page_name}`
    }
    eventApi.sendEvent(eventData)
  }

  componentDidMount() {
    var self = this;
    const ancillary = URLSearchParams.getParameterByName('ancillary', window.location.href)
    this.setState({ ancillary: ancillary ? ancillary : false })
    self.getFiltersValue()

    apiServices.getClassNameDetails().then((result) => {
      if (result) {
        var classData = result[0].classDatas;
        self.setState({ cabinOption: classData, cabinDisable: false })
      }
    });
  }

  componentDidUpdate() {
    window.onpopstate = e => {
      const obj = this.props.browserToProps.queryParams;
      let data = Object.values(obj);
      let title = Object.keys(obj);
      const lastIndex = title.length - 1
      if (data[0] !== 'undefined') {
        this.pushURLToBcData(obj, title, data, lastIndex)
        this.setState({ firstHome: true })
      } else {
        if (this.state.firstHome) {
          this.homeHandleClick();
        }
      }
    }
  }

  pushURLToBcData(obj, title, data, lastIndex) {
    const self = this;
    let region = []
    let country = []
    let city = []

    if (obj.hasOwnProperty('Region') && !bcData.some(function (o) { return o["title"] === "Region"; })) {
      let data = obj['Region']
      let bcContent = obj['Region'];
      let multiSelectLS;
      let regionId;

      if ((data).includes(',')) {
        data = `'${data.split(',').join("','")}'`;
      } else if (data.charAt(0) !== "'" && data.charAt(data.length - 1) !== "'") {
        data = `'${data}'`
      }

      if (bcContent.charAt(0) === "'" && bcContent.charAt(bcContent.length - 1) === "'") {
        regionId = bcContent.substring(1, bcContent.length - 1)
      } else if (bcContent.includes(',')) {
        multiSelectLS = bcContent.split(',');
        regionId = bcContent;
      }
      // else if(bcContent.includes("','")){
      //   multiSelectLS = bcContent.split("','");
      //   regionId = bcContent;
      // }
      else {
        regionId = bcContent;
      }
      
      bcData.push({ "val": regionId, "title": 'Region' });
      self.setState({ regionId: data })
      let regionLS = bcContent.includes(',') ? multiSelectLS : region.concat([regionId])
      window.localStorage.setItem('RegionSelected', JSON.stringify(regionLS))
    }
    if (obj.hasOwnProperty('Country') && !bcData.some(function (o) { return o["title"] === "Country"; })) {
      let data = obj['Country']
      let bcContent = obj['Country']
      let multiSelectLS;
      let countryId;

      if ((data).includes(',')) {
        data = `'${data.split(',').join("','")}'`;
      } else if (data.charAt(0) !== "'" && data.charAt(data.length - 1) !== "'") {
        data = `'${data}'`
      }
      if (bcContent.charAt(0) === "'" && bcContent.charAt(bcContent.length - 1) === "'") {
        countryId = bcContent.substring(1, bcContent.length - 1)
      } else if (bcContent.includes(',')) {
        multiSelectLS = bcContent.split(',');
        countryId = bcContent;
      } else {
        countryId = bcContent;
      }
      bcData.push({ "val": countryId, "title": 'Country' });
      self.setState({ countryId: data })
      let countryLS = bcContent.includes(',') ? multiSelectLS : country.concat([countryId])
      window.localStorage.setItem('CountrySelected', JSON.stringify(countryLS))

    }
    if (obj.hasOwnProperty('OD') && !bcData.some(function (o) { return o["title"] === "OD"; })) {
      let data = obj['OD']
      let bcContent = obj['OD']
      let multiSelectLS;
      let commonOD;

      if ((data).includes(',')) {
        data = `'${data.split(',').join("','")}'`;
      } else if (data.charAt(0) !== "'" && data.charAt(data.length - 1) !== "'") {
        data = `'${data}'`
      }
      if (bcContent.charAt(0) === "'" && bcContent.charAt(bcContent.length - 1) === "'") {
        commonOD = bcContent.substring(1, bcContent.length - 1)
      } else if (bcContent.includes(',')) {
        multiSelectLS = bcContent.split(',');
        commonOD = bcContent;
      } else {
        commonOD = bcContent;
      }

      bcData.push({ "val": commonOD, "title": 'OD' });
      self.setState({ commonOD: data })
      let commonODLS = bcContent.includes(',') ? multiSelectLS : city.concat([commonOD])
      window.localStorage.setItem('ODSelected', JSON.stringify(commonODLS))

    }
    if (obj.hasOwnProperty('O%26D') && !bcData.some(function (o) { return o["title"] === "CabinValue"; })) {

      bcData.push({ "val": obj['O%26D'], "title": 'CabinValue' });

      self.setState({ commonOD: obj['O%26D'] })
      window.localStorage.setItem('CabinValueSelected', obj['O%26D'])
    }

    if (bcData.length > 0) {
      var removeArrayIndex = bcData.slice(0, lastIndex + 1);
      bcData = removeArrayIndex;
    }

    this.listHandleClick(data[lastIndex], title[lastIndex], 'browserBack')
  }

  getFiltersValue = () => {
    bcData = []
    let RegionSelected = window.localStorage.getItem('RegionSelected')
    let CountrySelected = window.localStorage.getItem('CountrySelected')
    let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))
    let getCabinValue = window.localStorage.getItem('CabinSelected')
    let ODSelected = window.localStorage.getItem('ODSelected')
    let CabinSelected = window.localStorage.getItem('CabinValueSelected')
    let customerSegmentSelected = window.localStorage.getItem('CustomerSegmentSelected')

    let cabinSelectedDropDown = getCabinValue === null || getCabinValue === 'Null' ? [] : JSON.parse(getCabinValue);
    getCabinValue = cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : 'Null'


    this.setState({
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
      commonOD: ODSelected === null || ODSelected === 'Null' || ODSelected === '' || ODSelected === '*' ? '*' : JSON.parse(ODSelected),
      cabinId: CabinSelected === null || CabinSelected === 'Null' || CabinSelected === '' || CabinSelected === '*' ? '*' : JSON.parse(CabinSelected),
      customerSegmentationId: customerSegmentSelected === null || customerSegmentSelected === 'Null' || customerSegmentSelected === '' || customerSegmentSelected === '*' ? '*' : JSON.parse(customerSegmentSelected),
      gettingMonth: window.monthNumToName(rangeValue.from.month),
      gettingYear: rangeValue.from.year,
      getCabinValue: getCabinValue,
      cabinSelectedDropDown: cabinSelectedDropDown
    }, () => this.getInitialData())
  }

  getInitialData = () => {
    var self = this;
    let { gettingMonth, gettingYear, regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, getCabinValue, ancillary, enrich } = this.state;
    self.setState({ loading: true, loading2: true, firstLoadList: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    self.getInitialListData(regionId, countryId, commonOD, cabinId, customerSegmentationId);

    apiServices.getDemographyMonthTables(this.state.toggle, regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, getCabinValue).then(function (result) {
      self.setState({ loading: false, firstLoadList: false })
      if (result) {
        var totalData = result[0].totalData;
        var columnName = result[0].columnName;
        var posMonthdata = result[0].rowData;

        monthData = posMonthdata;

        self.setState({ posMonthDetails: self.getHighlightedMonth(posMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData })
      }

      if (ancillary) {
        self.getDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId, 'Ancillary');
        self.setState({ type: 'Ancillary' })
      } else {
        self.getDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId, 'Null');
      }

    });
  }

  getInitialListData = (regionId, countryId, commonOD, cabinId, customerSegmentationId) => {
    const self = this;
    const userDetails = JSON.parse(cookieStorage.getCookie('userDetails'));
    // let commonOD = OD.substring(1, OD.length - 1)
    let access = userDetails.access;
    let country = '*';
    let city = '*';

    if (access !== '#*') {
      self.setState({ accessLevelDisable: true })
      let accessList = access.split('#');
      country = accessList[2]
      city = accessList[2] === '*' ? '*' : accessList[3]
    }

    if (regionId !== '*') {
      bcData.push({ "val": regionId, "title": 'Region', 'disable': country !== '*' ? true : false });
      self.setState({ selectedData: regionId })
    }
    if (countryId !== '*') {
      bcData.push({ "val": countryId, "title": 'Country', 'disable': city !== '*' ? true : false });
      self.setState({ selectedData: countryId })
    }
    if (commonOD !== '*') {
      bcData.push({ "val": commonOD, "title": 'O&D' });
      self.setState({ selectedData: commonOD })
    }
    if (commonOD !== '*') {
      if (cabinId !== '*') {
        bcData.push({ "val": cabinId, "title": 'Cabin' });
        self.setState({ selectedData: cabinId })
      }
    }
    if (customerSegmentationId !== '*') {
      bcData.push({ "val": customerSegmentationId, "title": 'Customer Segment' });
      self.setState({ selectedData: customerSegmentationId })
    }
  }


  getMonthDrillDownData = (regionId, countryId, commonOD, cabinId, customerSegmentationId) => {
    var self = this;
    let { gettingMonth, enrichId, getCabinValue, type, gettingYear, toggle } = this.state;

    self.setState({ loading: true, loading2: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    apiServices.getDemographyMonthTables(toggle, regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, getCabinValue).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var totalData = result[0].totalData
        var columnName = result[0].columnName;
        var posMonthdata = result[0].rowData;
        monthData = posMonthdata;
        self.setState({ posMonthDetails: self.getHighlightedMonth(posMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData })
      }
    });

    apiServices.getDemographyDrillDownData(gettingYear, toggle, gettingMonth, regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: self.addCellRender(result[0].columnName),
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          commonOD: result[0].currentAccess.commonOD,
          cabinId: result[0].currentAccess.cabinId,
          customerSegmentationId: result[0].currentAccess.customerSegmentationId
        })
      }
    });
  }

  getDrillDownData = (regionId, countryId, commonOD, cabinId, customerSegmentationId, type) => {
    var self = this;
    let { gettingYear, gettingMonth, enrichId, getCabinValue, toggle } = this.state;

    apiServices.getDemographyDrillDownData(gettingYear, toggle, gettingMonth, regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: self.addCellRender(result[0].columnName),
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          commonOD: result[0].currentAccess.commonOD,
          cabinId: result[0].currentAccess.cabinId,
          customerSegmentationId: result[0].currentAccess.customerSegmentationId
        })
      }
    });
  }

  getHighlightedMonth(posMonthdata, month, year) {
    let monthNumber = window.monthNameToNum(month)
    let count = 0;
    let data = posMonthdata.filter((data, index) => {
      var monthName = data.Month;
      const selectedMonth = `${window.shortMonthNumToName(monthNumber)} ${year}`
      if (year !== currentYear - 1) {
        if (data.Year === currentYear - 1) {
          count = count + 1
        }
      }
      if (selectedMonth === monthName) {
        data.highlightMe = true;
        this.setState({ ensureIndexVisible: index - count })
      }
      if (year === currentYear) {
        if (data.Month === `Total ${currentYear - 1}`) {
          data.Month = `► Total ${currentYear - 1}`;
        }
        if (data.Month === `Total ${currentYear + 1}`) {
          data.Month = `► Total ${currentYear + 1}`;
        }
        if (data.Year === currentYear || data.Month === `► Total ${currentYear - 1}` || data.Month === `► Total ${currentYear + 1}`) {
          return data;
        }
      } else if (year === currentYear + 1) {
        if (data.Month === `Total ${currentYear - 1}`) {
          data.Month = `► Total ${currentYear - 1}`;
        }
        if (data.Month === `Total ${currentYear + 1}`) {
          data.Month = `▼ Total ${currentYear + 1}`;
        }
        if (data.Year >= currentYear || data.Month === `► Total ${currentYear - 1}` || data.Month === `▼ Total ${currentYear + 1}`) {
          return data;
        }
        this.setState({ showNextYearRows: true, showLastYearRows: false })
      } else if (year === currentYear - 1) {
        if (data.Month === `Total ${currentYear - 1}`) {
          data.Month = `▼ Total ${currentYear - 1}`;
        }
        if (data.Month === `Total ${currentYear + 1}`) {
          data.Month = `► Total ${currentYear + 1}`;
        }
        if (data.Year <= currentYear || data.Month === `▼ Total ${currentYear - 1}` || data.Month === `► Total ${currentYear + 1}`) {
          return data;
        }
        this.setState({ showLastYearRows: true, showNextYearRows: false })
      }
    })
    return data;
  }

  getHighlightedRow(updatedData, month) {
    let data = updatedData.map((data, index) => {
      let monthName = data.Month;
      if (monthName === `▼ Total ${currentYear - 1}` || monthName === `► Total ${currentYear - 1}`) {
        data.highlightMe = true;
      } else if (monthName === `▼ Total ${currentYear + 1}` || monthName === `► Total ${currentYear + 1}`) {
        data.highlightMe = true;
      }
      return data;
    })
    return data;
  }

  monthWiseCellClick = (params) => {
    var monththis = this;
    monththis.sendEvent('2', 'clicked on Months row', 'pos', 'Pos Page');
    let { gettingMonth, regionId, countryId, cityId, commonOD, cabinId, customerSegmentationId, getCabinValue, type, gettingYear } = this.state;
    let selectedMonth = params.data.Month;
    var column = params.colDef.field;
    let hyperLink = params.data.isUnderline

    const posMonthDetails = this.state.posMonthDetails.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: posMonthDetails });

    //Getting Clubbed Data
    if (selectedMonth.includes(`Total ${currentYear - 1}`)) {
      this.setState({ showLastYearRows: !this.state.showLastYearRows, showNextYearRows: false }, () => this.getLastYearClubbedData(this.state.showLastYearRows, selectedMonth))

    } else if (selectedMonth.includes(`Total ${currentYear + 1}`)) {
      this.setState({ showNextYearRows: !this.state.showNextYearRows, showLastYearRows: false }, () => this.getNextYearClubbedData(this.state.showNextYearRows, selectedMonth))

    } else {
      monththis.setState({ gettingMonth: params.data.MonthName, gettingYear: params.data.Year })
      const range = { from: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) }, to: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) } }
      window.localStorage.setItem('rangeValue', JSON.stringify(range))
    }

    if (column === 'CY_B' && !selectedMonth.includes('Total')) {
      
      params.event.stopPropagation();
      monththis.showLoader();
      apiServices.getPOSCabinDetails(params.data.Year, params.data.MonthName, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
        monththis.hideLoader();
        if (result) {
          var columnName = result[0].columnName;
          var cabinData = result[0].cabinData;
          monththis.setState({ tableModalVisible: true, modaldrillDownData: cabinData, modaldrillDownColumn: columnName })
        }
      });

    }
    // else if (column === 'FRCT/Act_P' && hyperLink) {
    //   params.event.stopPropagation();
    //   monththis.setState({ chartVisible: true, forecastChartHeader: 'Passenger Forecast' })

    // }
    // else if (column === 'FRCT/Act_A' && hyperLink) {
    //   params.event.stopPropagation();
    //   monththis.setState({ chartVisible: true, forecastChartHeader: 'Average fare Forecast' })

    // }
    // else if (column === 'FRCT/Act_R' && hyperLink) {
    //   params.event.stopPropagation();
    //   monththis.setState({ chartVisible: true, forecastChartHeader: 'Revenue Forecast' })
    // }
    // else if (column === 'CY_AL') {
    //   this.props.history.push('/topMarkets')

    // } 
    else if (column === 'Month' && !selectedMonth.includes('Total')) {
      monththis.setState({ loading2: true, drillDownData: [], drillDownTotalData: [] })
      monththis.getDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId, type)
    }
  }

  getLastYearClubbedData(showLastYearRows, selectedMonth) {
    if (showLastYearRows) {
      const updatedMonthData = monthData.filter((d) => {
        if (d.Month === `► Total ${currentYear - 1}`) {
          d.Month = `▼ Total ${currentYear - 1}`
        }
        if (d.Month === `▼ Total ${currentYear + 1}`) {
          d.Month = `► Total ${currentYear + 1}`
        }
        if (d.Year <= currentYear || d.Month === `▼ Total ${currentYear - 1}` || d.Month === `► Total ${currentYear + 1}`) {
          return d;
        }
      })
      this.setState({ posMonthDetails: this.getHighlightedRow(updatedMonthData, selectedMonth) })
    } else {
      const updatedMonthData = monthData.filter((d) => {
        if (d.Month === `▼ Total ${currentYear - 1}`) {
          d.Month = `► Total ${currentYear - 1}`
        }
        if (d.Month === `▼ Total ${currentYear + 1}`) {
          d.Month = `► Total ${currentYear + 1}`
        }
        if (d.Year === currentYear || d.Month === `► Total ${currentYear - 1}` || d.Month === `► Total ${currentYear + 1}`) {
          return d;
        }
      })
      this.setState({ posMonthDetails: this.getHighlightedRow(updatedMonthData, selectedMonth) })
    }
  }

  getNextYearClubbedData(showNextYearRows, selectedMonth) {
    if (showNextYearRows) {
      const updatedMonthData = monthData.filter((d) => {
        if (d.Month === `► Total ${currentYear + 1}`) {
          d.Month = `▼ Total ${currentYear + 1}`
        }
        if (d.Month === `▼ Total ${currentYear - 1}`) {
          d.Month = `► Total ${currentYear - 1}`
        }
        if (d.Year >= currentYear || d.Month === `▼ Total ${currentYear + 1}` || d.Month === `► Total ${currentYear - 1}`) {
          return d;
        }
      })
      this.setState({ posMonthDetails: this.getHighlightedRow(updatedMonthData, selectedMonth) })
    } else {
      const updatedMonthData = monthData.filter((d) => {
        if (d.Month === `▼ Total ${currentYear + 1}`) {
          d.Month = `► Total ${currentYear + 1}`
        }
        if (d.Month === `▼ Total ${currentYear - 1}`) {
          d.Month = `► Total ${currentYear - 1}`
        }
        if (d.Year === currentYear || d.Month === `► Total ${currentYear + 1}` || d.Month === `► Total ${currentYear - 1}`) {
          return d;
        }
      })
      this.setState({ posMonthDetails: this.getHighlightedRow(updatedMonthData, selectedMonth) })
    }
  }

  regionCellClick = (params) => {
    var self = this;
    self.sendEvent('2', 'clicked on Region drill down', 'pos', 'Pos Page');
    let { regionId, countryId, commonOD, cabinId, customerSegmentationId, getCabinValue } = this.state;

    var column = params.colDef.field;
    var selectedData = `'${params.data.firstColumnName}'`;
    var selectedDataWQ = params.data.firstColumnName;
    var selectedTitle = params.colDef.headerName
    let found;
    bcData.map((data, i) => data.title === selectedTitle ? found = true : found = false)

    // if (column === 'Avail') {
    //   if (!found) {
    //     if (regionId === '*') {
    //       self.getAvailabilityData(selectedData, countryId, cityId, commonOD, getCabinValue, 'Null', 'Null')

    //     } else if (countryId === '*') {
    //       self.getAvailabilityData(regionId, selectedData, cityId, commonOD, getCabinValue, 'Null', 'Null')

    //     } else if (cityId === '*') {
    //       self.getAvailabilityData(regionId, countryId, selectedData, commonOD, getCabinValue, 'Null', 'Null')

    //     } else if (commonOD === '*') {
    //       self.getAvailabilityData(regionId, countryId, cityId, selectedData, getCabinValue, 'Null', 'Null')

    //     } else {
    //       self.getAvailabilityData(regionId, countryId, cityId, commonOD, selectedData, 'Null', 'Null')
    //     }
    //   }
    // } else if (column === 'FRCT/Act_A') {
    //   this.getInfareGraphData(params)

    // } else if (column === 'CY_AL') {
    //   this.storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, selectedDataWQ);
    //   this.props.history.push('/topMarkets')

    // } else
    
    if (column === 'firstColumnName') {
      if (!found) {
        this.storeValuesToLS(regionId, countryId, commonOD, cabinId, customerSegmentationId, getCabinValue, selectedDataWQ);

        if (selectedTitle !== 'Cabin Category') {
          self.setState({ selectedData })
          bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
        }

        if (regionId === '*') {
          self.getMonthDrillDownData(selectedData, countryId, commonOD, cabinId, customerSegmentationId)

        } else if (countryId === '*') {
          self.getMonthDrillDownData(regionId, selectedData, commonOD, cabinId, customerSegmentationId)

        } else if (commonOD === '*') {
          self.getMonthDrillDownData(regionId, countryId, selectedData, cabinId, customerSegmentationId)

        } else if (cabinId === '*') {
          self.getMonthDrillDownData(regionId, countryId, commonOD, selectedData, customerSegmentationId)

        }
      }
    }
  }

  // getAvailabilityData(regionId, countryId, cityId, commonOD, getCabinValue, type, typeParam) {
  //   const { gettingYear, gettingMonth } = this.state;
  //   this.showLoader();
  //   apiServices.getAvailabilityDetails(gettingYear, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type, typeParam).then((result) => {
  //     this.hideLoader();
  //     if (result) {
  //       this.setState({
  //         availModalVisible: true,
  //         availModalColumn: result[0].columnName,
  //         availModalData: result[0].rowData,
  //       })
  //     }
  //   })
  // }

  rectifyURLValues(regionId, countryId, commonOD, cabinId, customerSegmentationId) {
    if (Array.isArray(regionId)) {
      this.selectedRegion = regionId.join(',')
    } else if (regionId.includes("','")) {
      this.selectedRegion = regionId.split("','").join(',')
      this.selectedRegion = this.selectedRegion.substring(1, this.selectedRegion.length - 1);
    } else {
      this.selectedRegion = regionId
      this.selectedRegion = this.selectedRegion.substring(1, this.selectedRegion.length - 1);
    }
    if (Array.isArray(countryId)) {
      this.selectedCountry = countryId.join(',')
    } else if (regionId.includes("','")) {
      this.selectedCountry = countryId.split("','").join(',')
      this.selectedCountry = this.selectedCountry.substring(1, this.selectedCountry.length - 1);
    } else {
      this.selectedCountry = countryId
      this.selectedCountry = this.selectedCountry.substring(1, this.selectedCountry.length - 1);
    }
    if (Array.isArray(commonOD)) {
      this.selectedOD = commonOD.join(',')
    } else if (regionId.includes("','")) {
      this.selectedOD = commonOD.split("','").join(',')
      this.selectedOD = this.selectedOD.substring(1, this.selectedOD.length - 1);
    } else {
      this.selectedOD = commonOD
      this.selectedOD = this.selectedOD.substring(1, this.selectedOD.length - 1);
    }
    if (Array.isArray(cabinId)) {
      this.selectedCabin = cabinId.join(',')
    } else if (regionId.includes("','")) {
      this.selectedCabin = cabinId.split("','").join(',')
      this.selectedCabin = this.selectedCabin.substring(1, this.selectedCabin.length - 1);
    } else {
      this.selectedCabin = cabinId
      this.selectedCabin = this.selectedCabin.substring(1, this.selectedCabin.length - 1);
    }
    // if (Array.isArray(customerSegmentationId)) {
    //   this.selectedCustomerSegment = customerSegmentationId.join(',')
    // } else if (regionId.includes("','")) {
    //   this.selectedCustomerSegment = customerSegmentationId.split("','").join(',')
    //   this.selectedCustomerSegment = this.selectedCustomerSegment.substring(1, this.selectedCustomerSegment.length - 1);
    // } else {
    //   this.selectedCustomerSegment = customerSegmentationId
    //   this.selectedCustomerSegment = this.selectedCustomerSegment.substring(1, this.selectedCustomerSegment.length - 1);
    // }
  }

  storeValuesToLS(regionId, countryId, commonOD, cabinId, getCabinValue, customerSegmentationId, data) {
    let region = []
    let country = []
    let od = []
    let cabinvalue = []
    let cabin = []
    let customerSegment = []

    this.rectifyURLValues(regionId, countryId, commonOD, cabinId, customerSegmentationId);

    if (regionId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(data)}`)
      region.push(data)
      window.localStorage.setItem('RegionSelected', JSON.stringify(region))

    } else if (countryId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${encodeURIComponent(data)}`)
      country.push(data)
      window.localStorage.setItem('CountrySelected', JSON.stringify(country))

    } else if (commonOD === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${encodeURIComponent(this.selectedCountry)}&${encodeURIComponent('O&D')}=${data}`)
      od.push(data)
      window.localStorage.setItem('ODSelected', JSON.stringify(od))

    } else if (cabinId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${encodeURIComponent(this.selectedCountry)}&${encodeURIComponent('O&D')}=${data}&cabin=${data}`)
      cabinvalue.push(data)
      window.localStorage.setItem('CabinValueSelected', JSON.stringify(cabinvalue))

    }
    else if (customerSegmentationId === '*') {
      this.props.history.push(`${this.pathName}?&customerSegment=${data}`)
      customerSegment.push(data)
      window.localStorage.setItem('CustomerSegmentSelected', JSON.stringify(customerSegment))

    } else if (getCabinValue === 'Null') {
      cabin.push(data)
      window.localStorage.setItem('CabinSelected', JSON.stringify(cabin))
    }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    self.sendEvent('2', `clicked on ${selectedType} tab`, 'pos', 'Pos Page');
    let { regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId, monthcolumns } = this.state;
    self.setState({ type: selectedType, drillDownData: [], drillDownTotalData: [], loading2: true })

    if (outerTab) {
      this.setState({ outerTab: true })
    } else {
      this.setState({ outerTab: false })
    }
    self.getDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId, selectedType)
    this.gridApiMonth.setColumnDefs(monthcolumns);
  }

  CustomercellClick = (params) => {
    var monththis = this;
    this.setState({
      customerSegmentationId_1: params.value
    })
    let { gettingMonth, toggle, regionId, countryId, commonOD, cabinId, customerSegmentationId, customerSegmentationId_1, getCabinValue, type, gettingYear, enrichId } = this.state;
    
    params.event.stopPropagation();
    monththis.showLoader();
    apiServices.getDemographySegmentationDetails(gettingYear, toggle, gettingMonth, regionId, countryId, commonOD, cabinId, customerSegmentationId, customerSegmentationId_1, enrichId, getCabinValue, type).then(function (result) {
    monththis.hideLoader();
    if (result) {
      var columnName = result[0].columnName;
      var segmentData = result[0].segmentData;
      monththis.setState({ tableModalVisible: true, modaldrillDownData: segmentData, modaldrillDownColumn: columnName })
    }
  });
    
  }

  NationalityCellClick = (params) => {
    var column = params.colDef.field;
    var { NationalityId } = this.state
    var selectedData = params.data.firstColumnName;

    if (column === 'firstColumnName') {
      window.localStorage.setItem('Nationality', selectedData)
      this.setState({ NationalityId: `'${selectedData}'` })
      this.setState({ chartVisible: true, clicked: true })
    }
  }

  AgeBandCellClick = (params) => {
    var column = params.colDef.field;
    var { AgeBandId } = this.state
    var selectedData = params.data.firstColumnName;

    if (column === 'firstColumnName') {
      this.setState({ AgeBandId: `'${selectedData}'` })
      this.setState({ chartVisible: true, clicked: true })
    }
  }

  homeHandleClick = (e) => {
    var self = this;
    const userDetails = JSON.parse(cookieStorage.getCookie('userDetails'));
    let access = userDetails.access;

    if (access === '#*') {
      self.sendEvent('2', 'clicked on Network', 'pos', 'Pos Page');

      self.setState({ loading: true, loading2: true, firstHome: false, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [], toggle: 'bc' })

      window.localStorage.setItem('RegionSelected', 'Null')
      window.localStorage.setItem('CountrySelected', 'Null')
      window.localStorage.setItem('ODSelected', 'Null')
      window.localStorage.setItem('CabinValueSelected', 'Null')
      window.localStorage.setItem('CustomerSegmentSelected', 'Null')
      window.localStorage.setItem('Nationality', 'Null')

      self.getMonthDrillDownData('*', '*', '*', '*', '*')

      bcData = [];
      var newURL = window.location.href.split("?")[0];
      window.history.pushState('object', document.title, newURL);
      // this.props.history.push('/pos')
    }
  }

  listHandleClick = (data, title, selection) => {
    var self = this;
    self.sendEvent('2', 'clicked on Drill down list', 'pos', 'Pos Page');
    let { regionId, countryId, commonOD, cabinId, customerSegmentationId } = this.state;
    var selectedData = data;
    if (selectedData.charAt(0) !== "'" && selectedData.charAt(selectedData.length - 1) !== "'") {
      selectedData = `'${data}'`
    }
    if ((data).includes(',')) {
      selectedData = `'${data.split(',').join("','")}'`;
    }
    self.setState({ selectedData, loading: true, loading2: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
    var getColName = decodeURIComponent(title);

    if (selection === 'List') {
      var indexEnd = bcData.findIndex(function (d) {
        return d.title == title;
      })
      var removeArrayIndex = bcData.slice(0, indexEnd + 1);
      bcData = removeArrayIndex;
      this.changeURLOnListClick(regionId, countryId, commonOD, cabinId, customerSegmentationId, data, getColName)

    } else if (selection === 'browserBack') {
      this.onBackPressClearLS(getColName)
    }
    
    if (getColName === 'Region') {
      self.getMonthDrillDownData(selectedData, '*', '*', '*')

    } else if (getColName === 'Country') {
      self.getMonthDrillDownData(regionId, selectedData, '*', '*')

    } else if (getColName === 'O&D') {
      self.getMonthDrillDownData(regionId, countryId, selectedData, '*')

    } else if (getColName === 'Cabin') {
      self.getMonthDrillDownData(regionId, countryId, commonOD, selectedData)
    }
  }

  changeURLOnListClick(regionId, countryId, commonOD, cabinId, customerSegmentationId, selectedData, getColName) {

    this.rectifyURLValues(regionId, countryId, commonOD, cabinId, customerSegmentationId);

    if (getColName === 'Region') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(selectedData)}`)
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');
      window.localStorage.setItem('CabinValueSelected', 'Null');

    } else if (getColName === 'Country') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${encodeURIComponent(selectedData)}`)
      window.localStorage.setItem('ODSelected', 'Null');
      window.localStorage.setItem('CabinValueSelected', 'Null');

    } else if (getColName === 'O&D') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${encodeURIComponent(this.selectedCountry)}&OD=${selectedData}`)
      window.localStorage.setItem('CabinValueSelected', 'Null');

    } else if (getColName === 'Cabin') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(this.selectedCountry)}&OD=${this.selectedOD}&${encodeURIComponent('cabin')}=${selectedData}`)
    }
  }

  onBackPressClearLS(getColName) {
    if (getColName === 'Region') {
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');
      window.localStorage.setItem('CabinValueSelected', 'Null');

    } else if (getColName === 'Country') {
      window.localStorage.setItem('ODSelected', 'Null');
      window.localStorage.setItem('CabinValueSelected', 'Null');

    } else if (getColName === 'OD') {
      window.localStorage.setItem('CabinValueSelected', 'Null');

    }
  }

  Enrichstatus = (e) => {
    let enrich = e.target.value;
    let { regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId } = this.state;
    this.sendEvent('2', 'clicked on Enrich Status', 'demograpgyAgeReport', 'demograpgyAgeReport Page');
    this.setState({ enrichId: enrich !== '*' ? `'${enrich}'` : enrich }, () => this.getMonthDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId, enrichId))
  }

  cabinSelectChange = (e) => {
    e.preventDefault();
    const getCabinValue = e.target.value;

    this.setState({
      getCabinValue: getCabinValue,
      cabinSelectedDropDown: getCabinValue,
    }, () => {
      window.localStorage.setItem('CabinSelected', JSON.stringify(getCabinValue));
    })
  }

  onCabinClose() {
    var self = this;
    self.sendEvent('2', 'clicked on Cabin drop down', 'pos', 'Pos Page');
    let { cabinSelectedDropDown } = this.state;

    if (cabinSelectedDropDown.length > 0) {
      this.getDataOnCabinChange()
    } else {
      this.setState({ getCabinValue: 'Null' }, () => this.getDataOnCabinChange())
      window.localStorage.setItem('CabinSelected', 'Null');
    }
  }

  getDataOnCabinChange() {
    var self = this;
    self.setState({
      loading: true, loading2: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: []
    })
    let { regionId, countryId, commonOD, cabinId, customerSegmentationId } = this.state;
    self.getMonthDrillDownData(regionId, countryId, commonOD, cabinId, customerSegmentationId)
  }

  // toggle = (e) => {
  //   let currency = e.target.value;
  //   let { regionId, countryId, cityId, commonOD } = this.state;
  //   this.sendEvent('2', 'clicked on Currency toggle', 'pos', 'Pos Page');
  //   this.setState({ toggle: currency }, () => this.getMonthDrillDownData(regionId, countryId, cityId, commonOD))
  // }

  redirection = (e) => {
    this.sendEvent('2', 'clicked on POS/Route drop down', 'pos', 'Pos Page');
    let name = e.target.value;
    let routeGroup = window.localStorage.getItem('RouteGroupSelected')
    routeGroup = routeGroup !== null ? JSON.parse(routeGroup).join(',') : 'Network'
    this.url = `/route?RouteGroup=${routeGroup}`;
    let regionId = window.localStorage.getItem('RouteRegionSelected')
    let countryId = window.localStorage.getItem('RouteCountrySelected')
    let routeId = window.localStorage.getItem('RouteSelected')
    let leg = window.localStorage.getItem('LegSelected')
    let flight = window.localStorage.getItem('FlightSelected')

    if (regionId !== null && regionId !== 'Null') {
      regionId = JSON.parse(regionId)
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}`
    }
    if (countryId !== null && countryId !== 'Null') {
      countryId = JSON.parse(countryId)
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}`
    }
    if (routeId !== null && routeId !== 'Null') {
      routeId = JSON.parse(routeId)
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}`
    }
    if (leg !== null && leg !== 'Null' && leg !== '') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}`
    }
    if (flight !== null && flight !== 'Null' && flight !== '') {
      this.url = `/route?RouteGroup=${routeGroup}&Region=${String.removeQuotes(regionId)}&Country=${String.removeQuotes(countryId)}&Route=${String.removeQuotes(routeId)}&Leg=${String.removeQuotes(leg)}&Flight=${String.removeQuotes(flight)}`
    }

    if (name === 'POS') {
      this.props.history.push('/pos')
      bcData = []
    } else {
      this.props.history.push(this.url)
      bcData = []
    }
  }

  addCellRender(columnDefs) {
    const a = columnDefs.map((c) => {
      if (c.field === 'firstColumnName') {
        c['cellRenderer'] = this.alerts
      }
      if (c.children) {
        c.children = this.addCellRender(c.children);
      }
      return c
    })
    return a;
  }

  alerts = (params) => {
    const value = params.value;
    const header = params.colDef.headerName;
    var element = document.createElement("span");
    let isAlert = params.data.isAlert;
    isAlert = isAlert !== undefined ? isAlert.toString() : null
    if (isAlert !== '0' && isAlert !== null) {
      // if (header !== 'Cabin' && header !== 'Agency' && header !== 'Ancillary') {
      var icon = document.createElement("i");
      icon.className = 'fa fa-bell'
      icon.onclick = (e) => {
        e.stopPropagation();
        this.getAlertCardsData(header, value)
      }
      element.appendChild(icon);
    }
    element.appendChild(document.createTextNode(params.value));
    return element;
  }

  getAlertCardsData = (header, value) => {
    const { gettingMonth, gettingYear, regionId, countryId, cityId, commonOD, getCabinValue } = this.state;
    const userData = JSON.parse(cookieStorage.getCookie('userDetails'))

    let region = Constant.addQuotesforMultiSelect(regionId)
    let country = Constant.addQuotesforMultiSelect(countryId)
    let city = Constant.addQuotesforMultiSelect(cityId)
    let cabin = Constant.addQuotesforMultiSelect(getCabinValue)
    let params = ``

    if (header !== 'Cabin') {
      params = `user_id=${userData.id}`
    }

    if (header === 'Region') {
      params = `${params}&regionId='${encodeURIComponent(value)}'`
    }
    if (header === 'Country') {
      params = `${params}&regionId=${(region)}&countryId='${encodeURIComponent(value)}'`
    }
    if (header === 'POS') {
      params = `${params}&regionId=${(region)}&countryId=${(country)}&cityId='${encodeURIComponent(value)}'`
    }
    if (header === 'O&D') {
      params = `${params}&regionId=${(region)}&countryId=${(country)}&cityId=${(city)}&commonOD='${encodeURIComponent(value)}'`
    }
    // if (header === 'Cabin') {
    //   params = `user_id'=${userData.id}&regionId=${(region)}&countryId=${(country)}&cityId=${(city)}&commonOD=${encodeURIComponent(commonOD)}`
    // }

    if (header !== 'Cabin') {
      api.get(`getallalerts?${params}&infare=low`)
        .then((response) => {
          if (response) {
            if (response.data.response.length > 0) {
              let data = response.data.response;
              this.setState({
                alertData: data,
                alertVisible: true
              })
            } else {
              this.setState({
                alertData: [],
                alertVisible: true
              })
            }
          }
        })
        .catch((err) => {

        })
    }
  }

  getInfareGraphData = (tableParams) => {
    const { gettingMonth, gettingYear, regionId, countryId, cityId, commonOD, getCabinValue, toggle } = this.state;
    let region = Constant.addQuotesforMultiSelect(regionId)
    let country = Constant.addQuotesforMultiSelect(countryId)
    let city = Constant.addQuotesforMultiSelect(cityId)
    let cabin = Constant.addQuotesforMultiSelect(getCabinValue)

    let header = tableParams.columnApi.columnController.columnDefs[0].children[0].headerName;
    let value = tableParams.data.firstColumnName

    let params = ``

    if (header !== 'Cabin') {
      params = `gettingMonth=${window.monthNameToNum(gettingMonth)}&getYear=${gettingYear}&getCabinValue=${cabin}&currency_params=${toggle}`
    }

    if (header === 'Region') {
      params = `${params}&regionId='${encodeURIComponent(value)}'`
    }
    if (header === 'Country') {
      params = `${params}&regionId=${(region)}&countryId='${encodeURIComponent(value)}'`
    }
    if (header === 'POS') {
      params = `${params}&regionId=${(region)}&countryId=${(country)}&cityId='${encodeURIComponent(value)}'`
    }
    if (header === 'O&D') {
      params = `${params}&regionId=${(region)}&countryId=${(country)}&cityId=${(city)}&commonOD='${encodeURIComponent(value)}'`
    }
    if (header === 'Cabin') {
      params = `currency_params=${toggle}&gettingMonth=${window.monthNameToNum(gettingMonth)}&getYear=${gettingYear}&getCabinValue='${value}'&regionId=${(region)}&countryId=${(country)}&cityId=${(city)}&commonOD=${encodeURIComponent(commonOD)}`
    }

    api.get(`infaregraph?${params}`)
      .then((response) => {
        if (response) {
          let graphData = response.data.response[0].GraphData;
          let currency = response.data.response[0].currencyData[0].Currency;
          if (graphData.length > 0) {
            this.setState({
              infareData: graphData,
              infareGraphHeading: `${value} (${currency})`,
              infareCurrency: currency,
              infareModalVisible: true
            })
          } else {
            this.setState({
              infareData: [],
              infareGraphHeading: `${value} (${currency})`,
              infareCurrency: currency,
              infareModalVisible: true
            })
          }
        }
      })
      .catch((err) => {
        Swal.fire({
          title: 'Error!',
          text: 'Something went wrong. Please try after some time',
          icon: 'error',
          confirmButtonText: 'Ok'
        }).then(() => {
          // window.location = '/'
          // cookieStorage.deleteCookie();
        })
      })
  }

  serialize = (params) => {
    var str = [];
    for (var p in params)
      if (params.hasOwnProperty(p)) {
        str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
      }
    return str.join("&");
  }

  showLoader = () => {
    $("#loaderImage").addClass("loader-visible")
  }

  hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible")
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
  }

  renderTabs = () => {
    let { tabName, outerTab, ancillary } = this.state;

    return (
      <ul className="nav nav-tabs" role="tablist">
        {tabName === 'O&D' || tabName === 'Cabin' || tabName === 'Cabin Category' || tabName === 'Nationality' || tabName === 'Age Band' || tabName === 'Enrich Tier' || tabName === 'Customer Segmentation' || tabName === 'Gender' ?
          <li role="presentation" onClick={() => this.tabClick('Sales Region', 'outerTab')}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              Sales Region
            </a>
          </li> : ''}

        {tabName === 'Cabin' || tabName === 'Cabin Category' || tabName === 'Nationality' || tabName === 'Age Band' || tabName === 'Enrich Tier' || tabName === 'Customer Segmention' || tabName === 'Gender' ?
          <li role="presentation" onClick={() => this.tabClick('OD', 'outerTab')}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              O&D
            </a>
          </li> : ''}

        {tabName === 'Cabin Category' || tabName === 'Nationality' || tabName === 'Age Band' || tabName === 'Enrich Tier' || tabName === 'Customer Segmention' || tabName === 'Gender' ?
          <li role="presentation" onClick={() => this.tabClick('OD', 'outerTab')}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              Cabin
            </a>
          </li> : ''}

        {tabName === 'Nationality' || tabName === 'Age Band' || tabName === 'Enrich Tier' || tabName === 'Customer Segmention' || tabName === 'Gender' ?
          <li role="presentation" onClick={() => this.tabClick('Cabin Category', 'outerTab')}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              Cabin Category
            </a>
          </li> : ''}

        <li role="presentation" className={`${ancillary ? '' : "active"}`} onClick={() => this.tabClick('Null')} >
          <a href="#Section1" aria-controls="home" role="tab" data-toggle="tab">
            {tabName}
          </a>
        </li>

        {tabName === 'O&D' ? '' : tabName === 'Cabin' ? '' : tabName === 'Cabin Category' ? '' :
          <li role="presentation" onClick={() => this.tabClick('OD')} className={outerTab ? 'active' : ''}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              O&D
            </a>
          </li>}

        {tabName === 'Cabin' ? '' : tabName === 'Cabin Category' ? '' :
          <li role="presentation" onClick={() => this.tabClick('Cabin')}>
            <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
              Cabin
            </a>
          </li>}
        {tabName === 'Cabin Category' ? '' :
          <li role="presentation" onClick={() => this.tabClick('Cabin Category')}>
            <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
              Cabin Category
            </a>
          </li>}
        <li role="presentation" className={`${ancillary ? 'active' : ""}`} onClick={() => this.tabClick('Customer Segmentation')}>
          <a href="#Section5" aria-controls="messages" role="tab" data-toggle="tab">
            Customer segmentation
          </a>
        </li>
        <li role="presentation" onClick={() => this.tabClick('Nationality')}>
          <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
            Nationality
          </a>
        </li>
        <li role="presentation" className={`${ancillary ? 'active' : ""}`} onClick={() => this.tabClick('Age Band')}>
          <a href="#Section4" aria-controls="messages" role="tab" data-toggle="tab">
            Age band
          </a>
        </li>
        <li role="presentation" className={`${ancillary ? 'active' : ""}`} onClick={() => this.tabClick('Enrich Tier')}>
          <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
            Enrich tier
          </a>
        </li>
        <li role="presentation" className={`${ancillary ? 'active' : ""}`} onClick={() => this.tabClick('Gender')}>
          <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
            Gender
          </a>
        </li>
        {/* <li role="presentation" onClick={() => this.props.history.push('/posPromotion')}>
          <a href="#Section5" aria-controls="messages" role="tab" data-toggle="tab">
            Promotion Dashboard
        </a>
        </li> */}

      </ul>
    )
  }

  renderInfareGraphModal() {
    return (
      <Modal
        show={this.state.infareModalVisible}
        onHide={() => this.setState({ infareModalVisible: false })}
        aria-labelledby="ModalHeader"
      >
        <Modal.Header closeButton>
          <Modal.Title id='ModalHeader'>{`Infare Graph of ${this.state.infareGraphHeading}`}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <InfareMultiLineChart infareData={this.state.infareData} currency={this.state.infareCurrency} />
        </Modal.Body>
      </Modal >
    )
  }

  renderAvailModal() {
    return (
      <DatatableModelDetails
        tableModalVisible={this.state.availModalVisible}
        rowData={this.state.availModalData}
        columns={this.state.availModalColumn}
        header={`Availability Details`}
        loading={this.state.loading3}
      />
    )
  }

  gridApiMonthly = (api) => {
    this.gridApiMonth = api;
  }

  render() {
    const { cabinOption, cabinSelectedDropDown, cabinDisable, accessLevelDisable, outerTab, ancillary } = this.state;
    return (
      <div className='pos-details'>
        <Loader />
        <TopMenuBar dashboardPath={'/demographyDashboard'} {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: '0px' }}>
              <div className="col-md-7 col-sm-7 col-xs-7 toggle1">
                {/* <select className="form-control cabinselect pos-route-dropdown" onChange={(e) => this.redirection(e)}>
                  <option value='POS' selected={true}>POS</option>
                  <option value='Route'>Route</option>
                </select> */}
                <h2>Demographic Report</h2>
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      <div style={{ cursor: accessLevelDisable ? 'not-allowed' : 'pointer' }}>
                        <li className={`${accessLevelDisable ? 'breadcrumb-disable' : ''}`} onClick={() => this.homeHandleClick()}> {'Network'} </li>
                      </div>
                      {this.state.firstLoadList ? "" : bcData.map((item) =>
                        <div style={{ cursor: item.disable ? 'not-allowed' : 'pointer' }}>
                          <li className={`${item.disable ? 'breadcrumb-disable' : ''}`} onClick={(e) => this.listHandleClick(e.target.id, item.title, 'List')} id={item.val} title={`${item.title} : ${item.val}`}>
                            {` > ${item.val}`}
                          </li>
                        </div>
                      )}
                    </ol>
                  </nav>
                </section>
              </div>

              <div className="col-md-5 col-sm-5 col-xs-5 toggle2" style={{ width: "750px" }}>
                <h4>Enrich Status :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.Enrichstatus(e)}>
                  <option value='*'>All</option>
                  <option value='Enrich'>Enrich</option>
                  <option value='Non Enrich'>Non-Enrich</option>
                </select>

                <div className='cabin-selection'>
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
                </div>

                {/* <h4>Select Currency :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.toggle(e)} disabled={this.state.countryId === '*' ? true : false}>
                  <option value='bc' selected={this.state.countryId === '*' || this.state.toggle === 'bc' ? true : false}>BC</option>
                  <option value='lc'>LC</option>
                </select> */}

              </div>

            </div>

          </div>
        </div>

        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12">
            <div className="x_panel" style={{ marginTop: "10px", height: 'calc(100vh - 130px)' }}>
              <div className="x_content">

                <DataTableComponent
                  gridApi={this.gridApiMonthly}
                  rowData={this.state.posMonthDetails}
                  columnDefs={this.state.monthcolumns}
                  loading={this.state.loading}
                  onCellClicked={(cellData) => this.monthWiseCellClick(cellData)}
                  frameworkComponents={{ customHeaderGroupComponent: POSCustomHeaderGroup }}
                  rowClassRules={this.state.posMonthRowClassRule}
                  pos={true}
                  ensureIndexVisible={this.state.ensureIndexVisible}
                />
                <TotalRow
                  rowData={this.state.monthTotalData}
                  columnDefs={this.state.monthcolumns}
                  loading={this.state.loading}
                  frameworkComponents={{ customHeaderGroupComponent: POSCustomHeaderGroup }}
                  responsive={true}
                  reducingPadding={true}
                />

                <div className="tab" id="posTableTab" role="tabpanel" style={{ marginTop: '10px' }}>

                  {this.renderTabs()}

                  <div className="tab-content tabs">
                    <div role="tabpanel" className={`tab-pane fade in ${ancillary ? '' : "active"}`} id="Section1">

                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.regionCellClick(cellData)}
                        loading={this.state.loading2}
                        pos={true}
                      />
                      <TotalRow
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                      />

                    </div>

                    <div role="tabpanel" className="tab-pane fade" id="Section2">

                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        loading={this.state.loading2}
                        pos={true}
                      />
                      <TotalRow
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                      />

                    </div>

                    <div role="tabpanel" className="tab-pane fade" id="Section3">
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.NationalityCellClick(cellData)}
                        loading={this.state.loading2}
                        pos={true}
                      />
                      <TotalRow
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    <div role="tabpanel" className="tab-pane fade" id="Section4">
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.AgeBandCellClick(cellData)}
                        loading={this.state.loading2}
                        pos={true}
                      />
                      <TotalRow
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
                        responsive={true}
                        reducingPadding={true}
                      />
                    </div>

                    <div role="tabpanel" className={`tab-pane fade in ${ancillary ? 'active' : ""}`} id="Section5">
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.CustomercellClick(cellData)}
                        loading={this.state.loading2}
                        pos={true}
                      />
                      <TotalRow
                        loading={this.state.loading2}
                        rowData={this.state.drillDownTotalData}
                        columnDefs={this.state.drillDownColumn}
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
          {/*<DatatableModelDetails
            tableModalVisible={this.state.tableModalVisible}
            rowData={this.state.modaldrillDownData}
            columns={this.state.modaldrillDownColumn}
            header={`${this.state.customerSegmentationId_1} - ${this.state.gettingMonth} ${this.state.gettingYear}` }
            loading={this.state.loading3}
          />*/}
          <DataTableModelDemoSegment
            tableModalVisible={this.state.tableModalVisible}
            rowData={this.state.modaldrillDownData}
            columns={this.state.modaldrillDownColumn}
            header={`${this.state.customerSegmentationId_1} - ${this.state.gettingMonth} ${this.state.gettingYear}` }
            loading={this.state.loading3}
          />
          <AlertModal
            alertVisible={this.state.alertVisible}
            alertData={this.state.alertData}
            closeAlertModal={() => this.setState({ alertVisible: false })}
          />

          {this.renderAvailModal()}

          {this.renderInfareGraphModal()}

          <DemographyGraphsPage
            chartVisible={this.state.chartVisible}
            gettingYear={this.state.gettingYear}
            gettingMonth={this.state.gettingMonth}
            regionId={this.state.regionId}
            countryId={this.state.countryId}
            commonOD={this.state.commonOD}
            cabinId={this.state.cabinId}
            enrichId={this.state.enrichId}
            getCabinValue={this.state.getCabinValue}
            clicked={this.state.clicked}
            type={this.state.type}
            NationalityId={this.state.NationalityId}
            AgeBandId={this.state.AgeBandId}
            closeChartModal={() => this.setState({ chartVisible: false, NationalityId: '*' })}
          />

        </div>

      </div>

    );
  }
}

const NewComponent = BrowserToProps(POSDetail);

export default NewComponent;