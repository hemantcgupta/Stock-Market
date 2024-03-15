import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import api from '../../API/api'
import AlertModal from '../../Component/AlertModal';
import InfareMultiLineChart from '../../Component/InfareMultiLineChart';
import DatatableModelDetails from '../../Component/dataTableModel';
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
import './PosDetails.scss';
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
      type: 'Null',
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
      ensureIndexVisible: null,
      startMonth: '',
      lastMonth: '',
      odsearchvalue: ''
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
      console.log('rahul Region', multiSelectLS)
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
      console.log('rahul Country', countryId, data)

    }
    if (obj.hasOwnProperty('POS') && !bcData.some(function (o) { return o["title"] === "POS"; })) {
      let data = obj['POS']
      let bcContent = obj['POS']
      let multiSelectLS;
      let cityId;

      if ((data).includes(',')) {
        data = `'${data.split(',').join("','")}'`;
      } else if (data.charAt(0) !== "'" && data.charAt(data.length - 1) !== "'") {
        data = `'${data}'`
      }
      if (bcContent.charAt(0) === "'" && bcContent.charAt(bcContent.length - 1) === "'") {
        cityId = bcContent.substring(1, bcContent.length - 1)
      } else if (bcContent.includes(',')) {
        multiSelectLS = bcContent.split(',');
        cityId = bcContent;
      } else {
        cityId = bcContent;
      }

      bcData.push({ "val": cityId, "title": 'POS' });
      self.setState({ cityId: data })
      let cityLS = bcContent.includes(',') ? multiSelectLS : city.concat([cityId])
      window.localStorage.setItem('CitySelected', JSON.stringify(cityLS))
      console.log('rahul POS', cityId, data)

    }
    if (obj.hasOwnProperty('O%26D') && !bcData.some(function (o) { return o["title"] === "O&D"; })) {

      bcData.push({ "val": obj['O%26D'], "title": 'O&D' });
      console.log('rahul OD', obj['O%26D'])

      self.setState({ commonOD: obj['O%26D'] })
      window.localStorage.setItem('ODSelected', obj['O%26D'])
    }

    console.log('rahul bcData before', bcData, lastIndex, data[lastIndex], obj)
    if (bcData.length > 0) {
      var removeArrayIndex = bcData.slice(0, lastIndex + 1);
      bcData = removeArrayIndex;
    }
    console.log('rahul bcData after', bcData)

    this.listHandleClick(data[lastIndex], title[lastIndex], 'browserBack')
  }

  getFiltersValue = () => {
    bcData = []
    let RegionSelected = window.localStorage.getItem('RegionSelected')
    let CountrySelected = window.localStorage.getItem('CountrySelected')
    let CitySelected = window.localStorage.getItem('CitySelected')
    let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))
    let getCabinValue = window.localStorage.getItem('CabinSelected')
    let ODSelected = window.localStorage.getItem('ODSelected')

    let cabinSelectedDropDown = getCabinValue === null || getCabinValue === 'Null' ? [] : JSON.parse(getCabinValue);
    getCabinValue = cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : 'Null'

    CitySelected = CitySelected === null || CitySelected === 'Null' || CitySelected === '' ? '*' : JSON.parse(CitySelected)

    this.setState({
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
      cityId: CitySelected,
      commonOD: ODSelected === null || ODSelected === 'Null' || ODSelected === '' || CitySelected === '*' ? '*' : JSON.parse(ODSelected),
      gettingMonth: window.monthNumToName(rangeValue.from.month),
      gettingYear: rangeValue.from.year,
      getCabinValue: getCabinValue,
      cabinSelectedDropDown: cabinSelectedDropDown
    }, () => this.getInitialData())
  }

  getInitialData = () => {
    var self = this;
    let { gettingMonth, gettingYear, regionId, countryId, cityId, commonOD, getCabinValue, ancillary } = this.state;
    self.setState({ loading: true, loading2: true, firstLoadList: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    self.getInitialListData(regionId, countryId, cityId, commonOD);

    apiServices.getPOSMonthTables(this.state.toggle, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
      self.setState({ loading: false, firstLoadList: false })
      if (result) {
        var totalData = result[0].totalData;
        var columnName = result[0].columnName;
        var posMonthdata = result[0].rowData;
        var startMonthValue = posMonthdata[1].Month
        var lastMonthValue = posMonthdata[posMonthdata.length - 1].Month

        monthData = posMonthdata;

        self.setState({ posMonthDetails: self.getHighlightedMonth(posMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData, startMonth: startMonthValue, lastMonth: lastMonthValue })
      }

      if (ancillary) {
        self.getDrillDownData(regionId, countryId, cityId, commonOD, 'Ancillary');
        self.setState({ type: 'Ancillary' })
      } else {
        self.getDrillDownData(regionId, countryId, cityId, commonOD, 'Null');
      }

    });
  }

  getInitialListData = (regionId, countryId, cityId, OD) => {
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
    if (cityId !== '*') {
      bcData.push({ "val": cityId, "title": 'POS' });
      self.setState({ selectedData: cityId })
    }
    if (cityId !== '*') {
      if (OD !== '*') {
        bcData.push({ "val": OD, "title": 'O&D' });
        self.setState({ selectedData: OD })
      }
    }
  }

  getMonthDrillDownData = (regionId, countryId, cityId, commonOD) => {
    var self = this;
    let { gettingMonth, getCabinValue, type, gettingYear, toggle, odsearchvalue } = this.state;

    self.setState({ loading: true, loading2: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    apiServices.getPOSMonthTables(toggle, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        console.log(result, 'jack')
        var totalData = result[0].totalData
        var columnName = result[0].columnName;
        var posMonthdata = result[0].rowData;
        monthData = posMonthdata;
        self.setState({ posMonthDetails: self.getHighlightedMonth(posMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData })
      }
    });

    apiServices.getPOSDrillDownData(gettingYear, toggle, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type, odsearchvalue).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: self.addCellRender(result[0].columnName),
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          cityId: result[0].currentAccess.cityId,
          commonOD: result[0].currentAccess.commonOD,
        })
      }
    });
  }

  getDrillDownData = (regionId, countryId, cityId, commonOD, type) => {
    var self = this;
    let { gettingYear, gettingMonth, getCabinValue, toggle, odsearchvalue } = this.state;

    apiServices.getPOSDrillDownData(gettingYear, toggle, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type, odsearchvalue).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: self.addCellRender(result[0].columnName),
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          cityId: result[0].currentAccess.cityId,
          commonOD: result[0].currentAccess.commonOD
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
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type, gettingYear, monthcolumns } = this.state;
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
      this.gridApiMonth.setColumnDefs(monthcolumns);

    } else if (selectedMonth.includes(`Total ${currentYear + 1}`)) {
      this.setState({ showNextYearRows: !this.state.showNextYearRows, showLastYearRows: false }, () => this.getNextYearClubbedData(this.state.showNextYearRows, selectedMonth))
      this.gridApiMonth.setColumnDefs(monthcolumns);

    } else {
      monththis.setState({ gettingMonth: params.data.MonthName, gettingYear: params.data.Year })
      const range = { from: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) }, to: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) } }
      window.localStorage.setItem('rangeValue', JSON.stringify(range))
      this.gridApiMonth.setColumnDefs(monthcolumns);
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
    else if (column === 'FRCT/Act_P' && hyperLink) {
      params.event.stopPropagation();
      monththis.setState({ chartVisible: true, forecastChartHeader: 'Passenger Forecast' })

    }
    else if (column === 'FRCT/Act_A' && hyperLink) {
      params.event.stopPropagation();
      monththis.setState({ chartVisible: true, forecastChartHeader: 'Average fare Forecast' })

    }
    else if (column === 'FRCT/Act_R' && hyperLink) {
      params.event.stopPropagation();
      monththis.setState({ chartVisible: true, forecastChartHeader: 'Revenue Forecast' })
    }
    else if (column === 'CY_AL') {
      this.props.history.push('/topMarkets')

    } else if (column === 'Month' && !selectedMonth.includes('Total')) {
      monththis.setState({ loading2: true, drillDownData: [], drillDownTotalData: [] })
      monththis.getDrillDownData(regionId, countryId, cityId, commonOD, type)
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
    let { regionId, countryId, cityId, commonOD, getCabinValue } = this.state;

    var column = params.colDef.field;
    var selectedData = `'${params.data.firstColumnName}'`;
    var selectedDataWQ = params.data.firstColumnName;
    var selectedTitle = params.colDef.headerName
    let found;
    bcData.map((data, i) => data.title === selectedTitle ? found = true : found = false)

    if (column === 'Avail') {
      if (!found) {
        if (regionId === '*') {
          self.getAvailabilityData(selectedData, countryId, cityId, commonOD, getCabinValue, 'Null', 'Null')

        } else if (countryId === '*') {
          self.getAvailabilityData(regionId, selectedData, cityId, commonOD, getCabinValue, 'Null', 'Null')

        } else if (cityId === '*') {
          self.getAvailabilityData(regionId, countryId, selectedData, commonOD, getCabinValue, 'Null', 'Null')

        } else if (commonOD === '*') {
          self.getAvailabilityData(regionId, countryId, cityId, selectedData, getCabinValue, 'Null', 'Null')

        } else {
          self.getAvailabilityData(regionId, countryId, cityId, commonOD, selectedData, 'Null', 'Null')
        }
      }
    } else if (column === 'FRCT/Act_A') {
      this.getInfareGraphData(params)

    } else if (column === 'CY_AL') {
      this.storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, selectedDataWQ);
      this.props.history.push('/topMarkets')

    } else if (column === 'firstColumnName') {
      if (!found) {
        this.storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, selectedDataWQ);

        if (selectedTitle !== 'Cabin') {
          self.setState({ selectedData })
          bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
        }

        if (regionId === '*') {
          self.getMonthDrillDownData(selectedData, countryId, cityId, commonOD)

        } else if (countryId === '*') {
          self.getMonthDrillDownData(regionId, selectedData, cityId, commonOD)

        } else if (cityId === '*') {
          self.getMonthDrillDownData(regionId, countryId, selectedData, commonOD)

        } else if (commonOD === '*') {
          self.getMonthDrillDownData(regionId, countryId, cityId, selectedData)
        }
      }
    }
  }

  getAvailabilityData(regionId, countryId, cityId, commonOD, getCabinValue, type, typeParam) {
    const { gettingYear, gettingMonth } = this.state;
    this.showLoader();
    apiServices.getAvailabilityDetails(gettingYear, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type, typeParam).then((result) => {
      this.hideLoader();
      if (result) {
        this.setState({
          availModalVisible: true,
          availModalColumn: result[0].columnName,
          availModalData: result[0].rowData,
        })
      }
    })
  }

  rectifyURLValues(regionId, countryId, cityId) {
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
    if (Array.isArray(cityId)) {
      this.selectedCity = cityId.join(',')
    } else if (regionId.includes("','")) {
      this.selectedCity = cityId.split("','").join(',')
      this.selectedCity = this.selectedCity.substring(1, this.selectedCity.length - 1);
    } else {
      this.selectedCity = cityId
      this.selectedCity = this.selectedCity.substring(1, this.selectedCity.length - 1);
    }
  }

  storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, data) {
    let region = []
    let country = []
    let city = []
    let od = []
    let cabin = []

    this.rectifyURLValues(regionId, countryId, cityId);


    if (regionId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(data)}`)
      region.push(data)
      window.localStorage.setItem('RegionSelected', JSON.stringify(region))

    } else if (countryId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(data)}`)
      country.push(data)
      window.localStorage.setItem('CountrySelected', JSON.stringify(country))

    } else if (cityId === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${data}`)
      city.push(data)
      window.localStorage.setItem('CitySelected', JSON.stringify(city))

    } else if (commonOD === '*') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${data}`)
      od.push(data)
      window.localStorage.setItem('ODSelected', JSON.stringify(od))

    } else if (getCabinValue === 'Null') {
      cabin.push(data)
      window.localStorage.setItem('CabinSelected', JSON.stringify(cabin))
    }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    self.sendEvent('2', `clicked on ${selectedType} tab`, 'pos', 'Pos Page');
    let { regionId, countryId, cityId, commonOD, monthcolumns } = this.state;
    self.setState({ type: selectedType, drillDownData: [], drillDownTotalData: [], loading2: true })

    if (outerTab) {
      this.setState({ outerTab: true })
    } else {
      this.setState({ outerTab: false })
    }
    self.getDrillDownData(regionId, countryId, cityId, commonOD, selectedType)
    this.gridApiMonth.setColumnDefs(monthcolumns);
  }

  ODcellClick = (params) => {
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, gettingYear } = this.state;

    var column = params.colDef.field;
    var selectedData = params.data.firstColumnName;

    if (column === 'CY_AL') {
      let od = []
      od.push(selectedData)
      window.localStorage.setItem('ODSelected', JSON.stringify(od))
      this.props.history.push('/topMarkets')
    } else if (column === 'Avail') {
      this.getAvailabilityData(regionId, countryId, cityId, commonOD, getCabinValue, 'OD', selectedData)
    }
  }

  agentCellClick = (params) => {
    var column = params.colDef.field;
    var selectedData = params.data.firstColumnName;

    if (column === 'firstColumnName') {
      window.localStorage.setItem('Agent', selectedData)
      window.location.href = '/agentAnalysis'
    }
  }

  compartmentCellClick = (params) => {
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, gettingYear } = this.state;

    var column = params.colDef.field;
    var selectedData = params.data.firstColumnName;

    if (column === 'Avail') {
      this.getAvailabilityData(regionId, countryId, cityId, commonOD, getCabinValue, 'Cabin', selectedData)
    }
    else if (column === 'FRCT/Act_A') {
      this.getInfareGraphData(params)
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
      window.localStorage.setItem('CitySelected', 'Null')
      window.localStorage.setItem('ODSelected', 'Null')

      self.getMonthDrillDownData('*', '*', '*', '*')

      bcData = [];
      var newURL = window.location.href.split("?")[0];
      window.history.pushState('object', document.title, newURL);
      // this.props.history.push('/pos')
    }
  }

  listHandleClick = (data, title, selection) => {
    var self = this;
    self.sendEvent('2', 'clicked on Drill down list', 'pos', 'Pos Page');
    let { regionId, countryId, cityId } = this.state;
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
      this.changeURLOnListClick(regionId, countryId, cityId, data, getColName)

    } else if (selection === 'browserBack') {
      this.onBackPressClearLS(getColName)
    }

    if (getColName === 'Region') {
      self.getMonthDrillDownData(selectedData, '*', '*', '*')

    } else if (getColName === 'Country') {
      self.getMonthDrillDownData(regionId, selectedData, '*', '*')

    } else if (getColName === 'POS') {
      self.getMonthDrillDownData(regionId, countryId, selectedData, '*')

    } else if (getColName === 'O&D') {
      self.getMonthDrillDownData(regionId, countryId, cityId, selectedData)
    }
  }

  changeURLOnListClick(regionId, countryId, cityId, selectedData, getColName) {

    this.rectifyURLValues(regionId, countryId, cityId);

    if (getColName === 'Region') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(selectedData)}`)
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'Country') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(selectedData)}`)
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'POS') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${selectedData}`)
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'O&D') {
      this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${selectedData}`)
    }
  }

  onBackPressClearLS(getColName) {
    if (getColName === 'Region') {
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'Country') {
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'POS') {
      window.localStorage.setItem('ODSelected', 'Null');

    }
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

  odsearchfilter = (e) => {
    const { regionId, countryId, cityId, commonOD, type, odsearchvalue } = this.state
    var ODvalue = e.target.value
    this.setState({ odsearchvalue: ODvalue })
    // var grid =  e.api.setQuickFilter(document.getElementById('filter-text-box').value);
    console.log(odsearchvalue, 'option')
    this.setState({ loading2: true })
    setTimeout(() => {
      this.getDrillDownData(regionId, countryId, cityId, commonOD, type)
    }, 4000);
  }

  getDataOnCabinChange() {
    var self = this;
    self.setState({
      loading: true, loading2: true, posMonthDetails: [], monthTotalData: [], drillDownData: [], drillDownTotalData: []
    })
    let { regionId, countryId, cityId, commonOD } = this.state;
    self.getMonthDrillDownData(regionId, countryId, cityId, commonOD)
  }

  toggle = (e) => {
    let currency = e.target.value;
    let { regionId, countryId, cityId, commonOD } = this.state;
    this.sendEvent('2', 'clicked on Currency toggle', 'pos', 'Pos Page');
    this.setState({ toggle: currency }, () => this.getMonthDrillDownData(regionId, countryId, cityId, commonOD))
  }

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
        {tabName === 'Cabin' ?
          <li role="presentation" onClick={() => this.tabClick('OD', 'outerTab')}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              O&D
            </a>
          </li> : ''}

        <li role="presentation" className={`${ancillary ? '' : "active"}`} onClick={() => this.tabClick('Null')} >
          <a href="#Section1" aria-controls="home" role="tab" data-toggle="tab">
            {tabName}
          </a>
        </li>

        {tabName === 'O&D' ? '' : tabName === 'Cabin' ? '' :
          <li role="presentation" onClick={() => this.tabClick('OD')} className={outerTab ? 'active' : ''}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              O&D
            </a>
          </li>}

        {tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('Cabin')}>
          <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
            Cabin
          </a>
        </li>}

        <li role="presentation" onClick={() => this.tabClick('Agency')}>
          <a href="#Section4" aria-controls="messages" role="tab" data-toggle="tab">
            Agency
          </a>
        </li>
        <li role="presentation" className={`${ancillary ? 'active' : ""}`} onClick={() => this.tabClick('Ancillary')}>
          <a href="#Section5" aria-controls="messages" role="tab" data-toggle="tab">
            Ancillary
          </a>
        </li>
        <li role="presentation" onClick={() => this.props.history.push('/posPromotion')}>
          <a href="#Section5" aria-controls="messages" role="tab" data-toggle="tab">
            Promotion Tracking
          </a>
        </li>

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
    const { cabinOption, cabinSelectedDropDown, cabinDisable, accessLevelDisable, outerTab, ancillary, startMonth, lastMonth } = this.state;
    return (
      <div className='pos-details'>
        <Loader />
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: '0px' }}>
              <div className="col-md-7 col-sm-7 col-xs-7 toggle1">
                <select className="form-control cabinselect pos-route-dropdown" onChange={(e) => this.redirection(e)}>
                  <option value='POS' selected={true}>POS</option>
                  <option value='Route'>Route</option>
                </select>
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

              <div>
                <h4> Data Available from {startMonth} to {lastMonth}</h4>
              </div>

              <div className="col-md-5 col-sm-5 col-xs-5 toggle2">
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

                <h4>Select Currency :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.toggle(e)} disabled={this.state.countryId === '*' ? true : false}>
                  <option value='bc' selected={this.state.countryId === '*' || this.state.toggle === 'bc' ? true : false}>BC</option>
                  <option value='lc'>LC</option>
                </select>

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

                      {this.state.cityId != '*' ?
                        <div style={{ height: '4vh', backgroundColor: '#1784c7' }}>
                          <input placeholder='Search OD'
                            onChange={this.odsearchfilter}
                            id="filter-text-box"
                            style={{ width: '6vw', height: '4vh' }}
                            value={this.state.odsearchvalue}
                          // disabled={this.state.cityId != '*' ? false : true}
                          >
                          </input>
                        </div>
                        : ''}

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

                      <div style={{ height: '4vh', backgroundColor: '#1784c7' }}>
                        <input placeholder='Search OD'
                          onChange={this.odsearchfilter}
                          id="filter-text-box"
                          style={{ width: '6vw', height: '4vh' }}
                          value={this.state.odsearchvalue}
                        // disabled={this.state.cityId != '*' ? false : true}
                        >
                        </input>
                      </div>

                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.ODcellClick(cellData)}
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
                        onCellClicked={(cellData) => this.compartmentCellClick(cellData)}
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
                        onCellClicked={(cellData) => this.agentCellClick(cellData)}
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
          <DatatableModelDetails
            tableModalVisible={this.state.tableModalVisible}
            rowData={this.state.modaldrillDownData}
            columns={this.state.modaldrillDownColumn}
            header={`${this.state.gettingMonth} ${this.state.gettingYear}`}
            loading={this.state.loading3}
          />
          <AlertModal
            alertVisible={this.state.alertVisible}
            alertData={this.state.alertData}
            closeAlertModal={() => this.setState({ alertVisible: false })}
          />

          {this.renderAvailModal()}

          {this.renderInfareGraphModal()}

          <ChartModelDetails
            chartVisible={this.state.chartVisible}
            displayName={this.state.forecastChartHeader}
            gettingYear={this.state.gettingYear}
            gettingMonth={this.state.gettingMonth}
            forecast={true}
            closeChartModal={() => this.setState({ chartVisible: false })}
          />

        </div>
        <p className='copyright-message'>Copyright © 2022 Revemax | All rights reserved | Contact</p>
      </div>

    );
  }
}

const NewComponent = BrowserToProps(POSDetail);

export default NewComponent;