import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import api from '../../API/api'
import AlertModal from '../../Component/AlertModal';
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
import '../../App';
import './ForecastAccuracy.scss';
import { indexOf } from 'lodash';


const apiServices = new APIServices();

const date = new Date();
const currmonthdate = new Date(date.getFullYear(),date.getMonth(),2).toISOString().slice(0,10);

const currentYear = new Date().getFullYear()
let monthData = [];
let bcData = [];

class ForecastAccuracy extends Component {
  constructor(props) {
    super(props);
    this.selectedRegion = null;
    this.selectedCountry = null;
    this.selectedCity = null;
    this.state = {
      monthData: [],
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
      currentMonth: (new Date()).getMonth()+1,
      currentYear: (new Date()).getFullYear(),
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
      currency: 'bc',
      alertVisible: false,
      tableModalVisible: false,
      tabName: 'Region',
      regionId: '*',
      countryId: '*',
      cityId: '*',
      commonOD: '*',
      type: 'Null',
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
      posMonthRowClassRule: {
        'highlight-row': 'data.highlightMe',
      },
      firstHome: true,
      infareCurrency: '',
      outerTab: false,
      ancillary: false,
      ensureIndexVisible: null
    }
    this.sendEvent('1', 'viewed Forecast Accuracy Page', 'forecastAccuracy', 'Forecast Accuracy Page');
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
    if (Access.accessValidation('Dashboard', 'forecastAccuracy')) {
      const ancillary = URLSearchParams.getParameterByName('ancillary', window.location.href)
      self.setState({ ancillary: ancillary ? ancillary : false })
      self.getFiltersValue()
    } else {
      self.props.history.push('/404')
    }

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
    const gettingMonth = window.monthNameToNum(this.getPrev2Months()[0].Month);
    const gettingYear = this.getPrev2Months()[0].Year;
    let RegionSelected = window.localStorage.getItem('')
    let CountrySelected = window.localStorage.getItem('')
    let CitySelected = window.localStorage.getItem('')
    let getCabinValue = window.localStorage.getItem('')

    let cabinSelectedDropDown = getCabinValue === null || getCabinValue === 'Null' ? [] : JSON.parse(getCabinValue);
    getCabinValue = cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : 'Null'

    CitySelected = CitySelected === null || CitySelected === 'Null' || CitySelected === '' ? '*' : JSON.parse(CitySelected)

    this.setState({
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
      startDate: Constant.formatDate(new Date(gettingYear, gettingMonth - 1, 1)),
      gettingMonth: gettingMonth,
      gettingYear: gettingYear,
      getCabinValue: getCabinValue,
      cabinSelectedDropDown: cabinSelectedDropDown
    }, () => this.getInitialData())
  }

  getInitialData = () => {
    var self = this;
    let { gettingMonth, gettingYear, regionId, countryId, cityId, commonOD, getCabinValue, startDate } = this.state;
    self.setState({ loading: true, loading2: true, firstLoadList: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    self.getInitialListData(regionId, countryId, cityId, commonOD);

    apiServices.getForecastMonthTable(startDate, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
      self.setState({ loading: false, firstLoadList: false })
      if (result) {
        var totalData = result[0].totalData;
        var columnName = result[0].columnName;
        var monthDatas = result[0].rowData;

        monthData = monthDatas;

        self.setState({ monthData: self.getHighlightedMonth(monthDatas, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData })
      }

      self.getDrillDownData(regionId, countryId, cityId, commonOD, 'Null');
    });
  }

  getInitialListData = (regionId, countryId, cityId, OD) => {
    const self = this;
    const userDetails = JSON.parse(cookieStorage.getCookie('userDetails'));
    let commonOD = OD.substring(1, OD.length - 1)
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
      if (commonOD !== '*') {
        bcData.push({ "val": commonOD, "title": 'O&D' });
        self.setState({ selectedData: OD })
      }
    }
  }

  getMonthDrillDownData = (regionId, countryId, cityId, commonOD) => {
    var self = this;
    let { gettingMonth, getCabinValue, type, gettingYear, startDate } = this.state;

    self.setState({ loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    apiServices.getForecastMonthTable(startDate, regionId, countryId, cityId, commonOD, getCabinValue).then(function (result) {
      self.setState({ loading: false })
      if (result) {
        var totalData = result[0].totalData
        var columnName = result[0].columnName;
        var monthDatas = result[0].rowData;
        monthData = monthDatas;
        self.setState({ monthData: self.getHighlightedMonth(monthDatas, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData })
      }
    });

    apiServices.getForecastDrillDownData(gettingYear, startDate, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: result[0].columnName,
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
    let { gettingYear, gettingMonth, getCabinValue, startDate } = this.state;

    apiServices.getForecastDrillDownData(gettingYear, startDate, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: result[0].columnName,
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          cityId: result[0].currentAccess.cityId,
          commonOD: result[0].currentAccess.commonOD
        })
      }
    });
  }

  getHighlightedMonth(monthDatas, month, year) {
    let monthNumber = month
    let current = new Date() 
    const selectedMonth = `${window.shortMonthNumToName(monthNumber)} ${year}`
    const Monthnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Sep','Oct','Nov','Dec']
    const currentMonth = Monthnames[current.getMonth()] + ' ' + current.getFullYear() 
    current.setMonth(current.getMonth()-1);
    const previousMoth = Monthnames[current.getMonth()] + ' ' + current.getFullYear() 
    let data = monthDatas.filter((data, index) => {
      var monthName = data.Month;
      if (selectedMonth === monthName) {
        data.highlightMe = true;
        this.setState({ ensureIndexVisible: index })
      }
      if (selectedMonth == currentMonth ){
        if (previousMoth == monthName){
        data.highlightMe = true;
        this.setState({ ensureIndexVisible: index })
      }
      }
      return data;
    })
  //}
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
    monththis.sendEvent('2', 'clicked on Months row', 'forecastAccuracy', 'Forecast Accuracy Page');
    let { regionId, countryId, cityId, commonOD, getCabinValue, type } = this.state;
    let selectedMonth = params.data.Month;
    var column = params.colDef.field;

    const monthData = this.state.monthData.map((d) => {
      d.highlightMe = false;
      return d;
    })
    params.api.updateRowData({ update: monthData });

    //Getting Clubbed Data
    if (selectedMonth.includes(`Total ${currentYear - 1}`)) {
      this.setState({ showLastYearRows: !this.state.showLastYearRows, showNextYearRows: false }, () => this.getLastYearClubbedData(this.state.showLastYearRows, selectedMonth))

    } else if (selectedMonth.includes(`Total ${currentYear + 1}`)) {
      this.setState({ showNextYearRows: !this.state.showNextYearRows, showLastYearRows: false }, () => this.getNextYearClubbedData(this.state.showNextYearRows, selectedMonth))

    } else {
      monththis.setState({ gettingMonth: window.monthNameToNum(params.data.MonthName), gettingYear: params.data.Year })
      const range = { from: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) }, to: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) } }
      window.localStorage.setItem('rangeValue', JSON.stringify(range))
    }

    if (column === 'Month' && !selectedMonth.includes('Total')) {
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
      this.setState({ monthData: this.getHighlightedRow(updatedMonthData, selectedMonth) })
      console.log("yash pos clubbing", updatedMonthData)
      console.log("yash pos new", this.state.monthData)
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
      this.setState({ monthData: this.getHighlightedRow(updatedMonthData, selectedMonth) })
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
      this.setState({ monthData: this.getHighlightedRow(updatedMonthData, selectedMonth) })
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
      this.setState({ monthData: this.getHighlightedRow(updatedMonthData, selectedMonth) })
    }
  }

  regionCellClick = (params) => {
    var self = this;
    self.sendEvent('2', 'clicked on Region drill down', 'forecastAccuracy', 'Forecast Accuracy Page');
    let {startDate, regionId, countryId, cityId, commonOD, gettingMonth, currentMonth, getCabinValue } = this.state;

    var column = params.colDef.field;
    var selectedData = `'${params.data.firstColumnName}'`;
    var selectedDataWQ = params.data.firstColumnName;
    var selectedTitle = params.colDef.headerName
    let found;
    const date = new Date();
    const currmonthdate = new Date(date.getFullYear(),date.getMonth(),2).toISOString().slice(0,10);
    bcData.map((data, i) => data.title === selectedTitle ? found = true : found = false)

    if (column === 'firstColumnName' && startDate !== currmonthdate) {
      if (!found) {
        if (selectedTitle !== 'POS') {
          self.setState({ selectedData })
          bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
          this.storeValuesToLS(regionId, countryId, cityId, commonOD, getCabinValue, selectedDataWQ);
        }

        if (regionId === '*') {
          self.getMonthDrillDownData(selectedData, countryId, cityId, commonOD)

        } else if (countryId === '*') {
          self.getMonthDrillDownData(regionId, selectedData, cityId, commonOD)

        }
        //  else if (cityId === '*') {
        //   self.getMonthDrillDownData(regionId, countryId, selectedData, commonOD)

        // } else if (commonOD === '*') {
        //   self.getMonthDrillDownData(regionId, countryId, cityId, selectedData)
        // }
      }
    }
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
    let cabin = []

    this.rectifyURLValues(regionId, countryId, cityId);

    if (regionId === '*') {
      this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(data)}`)
      region.push(data)
      window.localStorage.setItem('RegionSelected', JSON.stringify(region))

    } else if (countryId === '*') {
      this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(data)}`)
      country.push(data)
      window.localStorage.setItem('CountrySelected', JSON.stringify(country))

    }
    // else if (cityId === '*') {
    //   this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${data}`)
    //   city.push(data)
    //   window.localStorage.setItem('CitySelected', JSON.stringify(city))

    // } else if (commonOD === '*') {
    //   this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${data}`)
    //   window.localStorage.setItem('ODSelected', data)

    // }
    else if (getCabinValue === 'Null') {
      cabin.push(data)
      window.localStorage.setItem('CabinSelected', JSON.stringify(cabin))
    }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    self.sendEvent('2', `clicked on ${selectedType} tab`, 'forecastAccuracy', 'Forecast Accuracy Page');
    let { regionId, countryId, cityId, commonOD, selectedData } = this.state;
    self.setState({ type: selectedType, drillDownData: [], drillDownTotalData: [], loading2: true })

    if (outerTab) {
      this.setState({ outerTab: true })
    } else {
      this.setState({ outerTab: false })
    }
    self.getDrillDownData(regionId, countryId, cityId, commonOD, selectedType)
  }

  ODcellClick = (params) => {
    let { gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, gettingYear } = this.state;

    var column = params.colDef.field;
    var selectedData = params.data.firstColumnName;

    if (column === 'CY_AL') {
      window.localStorage.setItem('ODSelected', selectedData)
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
      self.sendEvent('2', 'clicked on Network', 'forecastAccuracy', 'Forecast Accuracy Page');

      self.setState({ loading: true, loading2: true, firstHome: false, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [], currency: 'bc' })

      window.localStorage.setItem('RegionSelected', 'Null')
      window.localStorage.setItem('CountrySelected', 'Null')
      window.localStorage.setItem('CitySelected', 'Null')
      window.localStorage.setItem('ODSelected', 'Null')

      self.getMonthDrillDownData('*', '*', '*', '*')

      bcData = [];
      var newURL = window.location.href.split("?")[0];
      window.history.pushState('object', document.title, newURL);
    }
  }

  listHandleClick = (data, title, selection) => {
    var self = this;
    self.sendEvent('2', 'clicked on Drill down list', 'forecastAccuracy', 'Forecast Accuracy Page');
    let { regionId, countryId, cityId } = this.state;
    var selectedData = data;
    if (selectedData.charAt(0) !== "'" && selectedData.charAt(selectedData.length - 1) !== "'") {
      selectedData = `'${data}'`
    }
    if ((data).includes(',')) {
      selectedData = `'${data.split(',').join("','")}'`;
    }
    self.setState({ selectedData, loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
    var getColName = decodeURIComponent(title);

    if (selection === 'List') {
      var indexEnd = bcData.findIndex(function (d) {
        return d.val == data;
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

    }
    //  else if (getColName === 'POS') {
    //   self.getMonthDrillDownData(regionId, countryId, selectedData, '*')

    // } else if (getColName === 'O&D') {
    //   self.getMonthDrillDownData(regionId, countryId, cityId, selectedData)
    // }
  }

  changeURLOnListClick(regionId, countryId, cityId, selectedData, getColName) {

    this.rectifyURLValues(regionId, countryId, cityId);

    if (getColName === 'Region') {
      this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(selectedData)}`)
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'Country') {
      this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(selectedData)}`)
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    }
    //  else if (getColName === 'POS') {
    //   this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${selectedData}`)
    //   window.localStorage.setItem('ODSelected', 'Null');

    // } else if (getColName === 'O&D') {
    //   this.props.history.push(`/forecastAccuracy?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${selectedData}`)
    // }
  }

  onBackPressClearLS(getColName) {
    if (getColName === 'Region') {
      window.localStorage.setItem('CountrySelected', 'Null');
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    } else if (getColName === 'Country') {
      window.localStorage.setItem('CitySelected', 'Null');
      window.localStorage.setItem('ODSelected', 'Null');

    }
    // else if (getColName === 'POS') {
    //   window.localStorage.setItem('ODSelected', 'Null');

    // }
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
    self.sendEvent('2', 'clicked on Cabin drop down', 'forecastAccuracy', 'Forecast Accuracy Page');
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
      loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: []
    })
    let { regionId, countryId, cityId, commonOD } = this.state;
    self.getMonthDrillDownData(regionId, countryId, cityId, commonOD)
  }

  toggle = (e) => {
    let { regionId, countryId, cityId, commonOD } = this.state;
    let currency = e.target.value;
    this.sendEvent('2', 'clicked on Currency toggle', 'forecastAccuracy', 'Forecast Accuracy Page');
    this.setState({ currency: currency }, () => this.getMonthDrillDownData(regionId, countryId, cityId, commonOD))
  }

  monthChange = () => {
    let { regionId, countryId, cityId, commonOD, currentMonth } = this.state;
    const e = document.getElementById('months')
    const gettingMonth = window.monthNameToNum(e.options[e.selectedIndex].id);
    const gettingYear = e.options[e.selectedIndex].value;
    let startDate = Constant.formatDate(new Date(gettingYear, gettingMonth - 1, 1))
    const iscurrentMonth = gettingMonth === currentMonth;
    bcData = [];
    this.setState({
      gettingMonth: gettingMonth,
      gettingYear: gettingYear,
      startDate: startDate,
      countryId: iscurrentMonth ? '*' : countryId,
      regionId: iscurrentMonth ? '*' : regionId,
      cityId: iscurrentMonth ? '*' : cityId,
      commonOD: iscurrentMonth ? '*' : commonOD,
    }, () => this.getMonthDrillDownData('*', '*', '*', '*'))
  }

  getPrev2Months() {
    const { currentMonth, currentYear } = this.state;
    let monthsArray = [...Array(currentMonth)].map((d,n) => (currentMonth - n) % 12)// %12 caters for end of year wrap-around.
    console.log(monthsArray)
    const prev2Months = monthsArray.map((d, i) => {
      let year = currentYear;
     /* if (d < 0) {
        d = 12 + d
        year = currentYear - 1 
      }
      if (d === 0) {
       d = 12
       year = currentYear - 1
      }
      if (monthsArray[0] === -1 && monthsArray[1] === 0) {
       year = currentYear - 1
      }
      if (monthsArray[i + 0] === 0) {
       year = currentYear - 1
      }*/
    return { 'MonthShort': window.shortMonthNumToName(d), 'Month': window.monthNumToName(d), 'Year': year };
    })
    return prev2Months;
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
    let {startDate, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, tabName, gettingYear, type, outerTab, ancillary, currentMonth } = this.state;
    const downloadURLDrillDown = apiServices.exportCSVPOSDrillDownURL(gettingYear, gettingMonth, regionId, countryId, cityId, commonOD, getCabinValue, type)
    const downloadURLMonthly = apiServices.exportCSVPOSMonthlyURL(regionId, countryId, cityId, commonOD, getCabinValue)

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

        {/* {tabName === 'O&D' ? '' : tabName === 'Cabin' ? '' :
          <li role="presentation" onClick={() => this.tabClick('OD')} className={outerTab ? 'active' : ''}>
            <a href="#Section2" aria-controls="profile" role="tab" data-toggle="tab">
              O&D
         </a>
          </li>} */}

        {startDate === currmonthdate || tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('Cabin')}>
          <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
            Cabin
        </a>
        </li>}

        {/* <DownloadCSV url={downloadURLDrillDown} name={`POS DRILLDOWN`} path={`/pos`} page={`Forecast Accuracy Page`} />
        <DownloadCSV url={downloadURLMonthly} name={`POS MONTHLY`} path={`/pos`} page={`Forecast Accuracy Page`} /> */}

      </ul>
    )
  }

  render() {
    const { cabinOption, cabinSelectedDropDown, cabinDisable, accessLevelDisable, outerTab, ancillary } = this.state;
    return (
      <div className='forecastAccuracy'>
        <Loader />
        <TopMenuBar {...this.props} />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: '0px' }}>
              <div className="col-md-7 col-sm-7 col-xs-7 toggle1">
                <h2>Forecast Accuracy</h2>
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


                <h4>Select Month :</h4>
                <select className="form-control cabinselect currency-dropdown" id="months" onChange={(e) => this.monthChange(e)}>
                  {this.getPrev2Months().map((d) => <option value={d.Year} id={d.Month}>{`${d.MonthShort} ${d.Year}`}</option>)}
                </select>

                {/* <h4>Select Currency :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.toggle(e)} disabled={this.state.countryId === '*' ? true : false}>
                  <option value='bc' selected={this.state.countryId === '*' || this.state.currency === 'bc' ? true : false}>BC</option>
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
                  rowData={this.state.monthData}
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

                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div>
          <AlertModal
            alertVisible={this.state.alertVisible}
            alertData={this.state.alertData}
            closeAlertModal={() => this.setState({ alertVisible: false })}
          />
        </div>

      </div>

    );
  }
}

const NewComponent = BrowserToProps(ForecastAccuracy);

export default NewComponent;