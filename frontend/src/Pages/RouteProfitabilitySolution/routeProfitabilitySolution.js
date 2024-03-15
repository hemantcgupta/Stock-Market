import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import ChartModelDetails from '../../Component/chartModel'
import DatatableModelDetails from '../../Component/dataTableModel'
import DataTableComponent from '../../Component/DataTableComponent'
import DownloadCSV from '../../Component/DownloadCSV';
import Loader from '../../Component/Loader';
import TotalRow from '../../Component/TotalRow';
import color from '../../Constants/color'
import $, { data } from 'jquery';
import '../../App.scss';
import './RouteProfitabilitySolution.scss';
import TopMenuBar from '../../Component/TopMenuBar';
import RouteCustomHeaderGroup from './RouteCustomHeaderGroup';
import cookieStorage from '../../Constants/cookie-storage';
import BrowserToProps from 'react-browser-to-props';
import Constant from '../../Constants/validator';
import { da } from 'date-fns/locale';
import { isNumber } from 'lodash';


const apiServices = new APIServices();


const currentYear = new Date().getFullYear()
let originalMonthData = [];

let bcData = [];
let baseAccess = '';

class RouteProfitabilitySolution extends Component {
  constructor(props) {
    super(props);
    this.userDetails = JSON.parse(cookieStorage.getCookie('userDetails'))
    this.selectedRegion = null;
    this.selectedCountry = null;
    this.selectedRoute = null;
    this.gridApiMonth = null;
    this.state = {
      showLastYearRows: false,
      routeMonthRowClassRule: {
        'highlight-row': 'data.highlightDay',
        'highlight-dBlue-row': 'data.highlightMonth'
      },
      monthData: [],
      routeMonthColumns: [],
      monthTotalData: [],
      ensureIndexVisible: null,
      drillDownTotalData: [],
      drillDownData: [],
      drillDownColumn: [],
      modelRegionDatas: [],
      modelregioncolumn: [],
      modalData: [],
      modalCompartmentColumn: [],
      modalCompartmentData: [],
      tableDatas: true,
      gettingDay: null,
      gettingMonth: null,
      gettingYear: null,
      monthTableTitle: '',
      tableTitle: '',
      tabLevel: '',
      cabinOption: [],
      getCabinValue: [],
      cabinSelectedDropDown: [],
      cabinDisable: true,
      // currency: 'bc',
      currency: 'vc',
      aircraft: 'all',
      chartVisible: false,
      tableModalVisible: false,
      tabName: 'Region',
      regionId: '*',
      countryId: '*',
      routeId: '*',
      leg: '*',
      flight: '*',
      type: 'Null',
      baseAccess: '',
      routeGroup: '',
      accessLevelDisable: false,
      selectedData: 'Null',
      loading: false,
      loading2: false,
      loading3: false,
      firstLoadList: false,
      firstHome: true,
      posContributionModal: false,
      regionLevelAccess: false,
      count: 0
    }
    Constant.sendEvent('1', 'viewed Route Profitability Solution Page', '/routeProfitabilitySolution', 'Route Profitability Solution');
  }

  componentDidMount() {
    var self = this;
    const aircraft = Constant.getParameterByName('aircraft', window.location.href)
    const route = Constant.getParameterByName('route', window.location.href)
    this.setState({ aircraftParam: aircraft ? aircraft : false, routeParam: route ? route : false })
    self.getFiltersValue();
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
      this.setState({ backPress: true })
    }
  }

  pushURLToBcData(obj, title, data, lastIndex) {
    const self = this;
    let group = []
    let region = []
    let country = []
    let city = []

    let routeGroup = obj['RouteGroup']
    this.setState({ routeGroup: routeGroup })
    window.localStorage.setItem('RouteGroupSelected', JSON.stringify(group.concat([routeGroup])))

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
      } else {
        regionId = bcContent;
      }

      bcData.push({ "val": regionId, "title": 'Region' });
      self.setState({ regionId: data })
      let regionLS = bcContent.includes(',') ? multiSelectLS : region.concat([regionId])
      window.localStorage.setItem('RouteRegionSelected', JSON.stringify(regionLS))
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
      window.localStorage.setItem('RouteCountrySelected', JSON.stringify(countryLS))
      console.log('rahul Country', countryId, data)

    }
    if (obj.hasOwnProperty('Route') && !bcData.some(function (o) { return o["title"] === "Route"; })) {
      let data = obj['Route']
      let bcContent = obj['Route']
      let multiSelectLS;
      let routeId;

      if ((data).includes(',')) {
        data = `'${data.split(',').join("','")}'`;
      } else if (data.charAt(0) !== "'" && data.charAt(data.length - 1) !== "'") {
        data = `'${data}'`
      }
      if (bcContent.charAt(0) === "'" && bcContent.charAt(bcContent.length - 1) === "'") {
        routeId = bcContent.substring(1, bcContent.length - 1)
      } else if (bcContent.includes(',')) {
        multiSelectLS = bcContent.split(',');
        routeId = bcContent;
      } else {
        routeId = bcContent;
      }

      bcData.push({ "val": routeId, "title": 'Route' });
      self.setState({ routeId: data })
      let cityLS = bcContent.includes(',') ? multiSelectLS : city.concat([routeId])
      window.localStorage.setItem('RouteSelected', JSON.stringify(cityLS))
      console.log('rahul Route', routeId, data)

    }
    if (obj.hasOwnProperty('Flight') && !bcData.some(function (o) { return o["title"] === "Flight"; })) {

      bcData.push({ "val": obj['Flight'], "title": 'Flight' });
      console.log('rahul Flight', obj['Flight'])

      self.setState({ flight: obj['Flight'] })
      window.localStorage.setItem('RPFlightSelected', obj['Flight'])
    }

    console.log('rahul bcData before', bcData, lastIndex)

    if (bcData.length > 0) {
      var removeArrayIndex = bcData.slice(0, lastIndex);
      bcData = removeArrayIndex;
    }
    console.log('rahul bcData after', bcData)


    this.listHandleClick(data[lastIndex], title[lastIndex], 'browserBack')
  }

  getFiltersValue = () => {
    bcData = []
    let routeGroup = window.localStorage.getItem('RouteGroupSelected')
    let RegionSelected = window.localStorage.getItem('RouteRegionSelected')
    let CountrySelected = window.localStorage.getItem('RouteCountrySelected')
    let RouteSelected = window.localStorage.getItem('RouteSelected')
    let RRDateRange = JSON.parse(window.localStorage.getItem('RRDateRangeValue'))
    let endDateArray = RRDateRange[0].endDate.split('-')
    let getCabinValue = window.localStorage.getItem('CabinSelected')
    let FlightSelected = window.localStorage.getItem('RPFlightSelected')

    if (routeGroup === null || routeGroup === '' || routeGroup === 'Null') {
      if (Object.keys(this.userDetails.route_access).length > 0) {
        routeGroup = this.userDetails.route_access['selectedRouteGroup']
      } else {
        routeGroup = ['Network']
      }
    } else {
      routeGroup = JSON.parse(routeGroup)
    }

    let cabinSelectedDropDown = getCabinValue === null || getCabinValue === 'Null' ? [] : JSON.parse(getCabinValue);
    getCabinValue = cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : 'Null'

    this.setState({
      routeGroup: routeGroup.join("','"),
      regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
      countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
      routeId: RouteSelected === null || RouteSelected === 'Null' || RouteSelected === '' ? '*' : JSON.parse(RouteSelected),
      flight: FlightSelected === null || FlightSelected === 'Null' || FlightSelected === '' ? '*' : `'${FlightSelected}'`,
      gettingYear: endDateArray[0],
      gettingMonth: window.monthNumToName(endDateArray[1]),
      gettingDay: RRDateRange[0].withoutDropDown ? null : endDateArray[2],
      getCabinValue: getCabinValue,
      cabinSelectedDropDown: cabinSelectedDropDown,
      count: RRDateRange[0].withoutDropDown ? 1 : 0
    }, () => this.getInitialData())
  }

  getInitialData() {
    let self = this;
    let { routeGroup, currency, regionId, countryId, routeId, flight, getCabinValue, aircraftParam, routeParam } = this.state;

    self.setState({ loading: true, loading2: true, firstLoadList: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

    self.getInitialListData(regionId, countryId, routeId, flight);

    apiServices.getRouteProfitabilitySolutionMonthlyData(currency, routeGroup, regionId, countryId, routeId, flight, getCabinValue).then((result) => {
      self.setState({ firstLoadList: false })
      let isCollapseMonth = Constant.isOdd(this.state.count) ? true : false;
      if (result) {
        originalMonthData = result[0].rowData;
        self.setState({
          routeMonthColumns: result[0].columnName,
          monthTotalData: result[0].totalData,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          routeId: result[0].currentAccess.routeId,
          flight: result[0].currentAccess.flight,
        }, () => self.getDayLevelData('', isCollapseMonth))
      }
    })

    if (aircraftParam) {
      self.getDrillDownData(routeGroup, regionId, countryId, routeId, flight, 'Aircraft');
      self.setState({ type: 'Aircraft' })
    } else if (routeParam) {
      self.getDrillDownData(routeGroup, regionId, countryId, routeId, flight, 'OD');
      self.setState({ type: 'OD' })
    } else {
      self.getDrillDownData(routeGroup, regionId, countryId, routeId, flight, 'Null');
    }

  }

  getInitialListData = (regionId, countryId, routeId, FLIGHT) => {
    const self = this;
    const routeAccess = this.userDetails.route_access
    let flight = FLIGHT.substring(1, FLIGHT.length - 1)

    if (Object.keys(routeAccess).length > 0) {
      self.setState({ accessLevelDisable: true })
    }
    const regionLevelAccess = (routeAccess).hasOwnProperty('selectedRouteRegion')
    self.setState({ regionLevelAccess })
    const countryLevelAccess = (routeAccess).hasOwnProperty('selectedRouteCountry')
    const routeLevelAccess = (routeAccess).hasOwnProperty('selectedRoute')

    if (regionId !== '*') {
      bcData.push({ "val": regionId, "title": 'Region', 'disable': countryLevelAccess });
      self.setState({ selectedData: regionId })
    }
    if (countryId !== '*') {
      bcData.push({ "val": countryId, "title": 'Country', 'disable': routeLevelAccess });
      self.setState({ selectedData: countryId })
    }
    if (routeId !== '*') {
      bcData.push({ "val": routeId, "title": 'Route' });
      self.setState({ selectedData: routeId })
    }
    if (flight !== '*') {
      bcData.push({ "val": flight, "title": 'Flight' });
      self.setState({ selectedData: FLIGHT })
    }
  }

  getMonthDrillDownData = (routeGroup, regionId, countryId, routeId, flight) => {
    var self = this;
    let { currency, gettingDay, gettingMonth, getCabinValue, type, gettingYear } = this.state;
    self.setState({ loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
    let isCollapseMonth = Constant.isOdd(this.state.count) ? true : false;
    apiServices.getRouteProfitabilitySolutionMonthlyData(currency, routeGroup, regionId, countryId, routeId, flight, getCabinValue).then((result) => {
      if (result) {
        originalMonthData = result[0].rowData;
        self.setState({
          routeMonthColumns: result[0].columnName,
          monthTotalData: result[0].totalData,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          routeId: result[0].currentAccess.routeId,
          flight: result[0].currentAccess.flight,
        }, () => self.getDayLevelData('', isCollapseMonth))
      }
    });

    apiServices.getRouteProfitabilitySolutionDrillDownData(gettingYear, currency, gettingMonth, gettingDay, routeGroup, regionId, countryId, routeId, flight, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: result[0].columnName,
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          routeId: result[0].currentAccess.routeId,
          flight: result[0].currentAccess.flight,
        })
      }
    });
  }

  getDayLevelData(click, isCollapseMonth) {
    let { routeGroup, currency, gettingDay, gettingMonth, gettingYear, regionId, countryId, routeId, flight, getCabinValue } = this.state;
    let _ = click === 'monthClick' ? this.showLoader() : null;
    apiServices.getRouteProfitabilitySolutionDayData(currency, gettingYear, gettingMonth, routeGroup, regionId, countryId, routeId, flight, getCabinValue).then((result) => {
      this.setState({ loading: false })
      this.hideLoader()
      if (result) {
        var dayLevelData = result[0].rowData;
        this.setState({
          monthData: isCollapseMonth ? this.getCollapsedMonth(originalMonthData, gettingMonth, gettingYear) : this.getExpandedMonth(dayLevelData, gettingDay, gettingMonth, gettingYear, isCollapseMonth)
        })
        this.gridApiMonth.ensureIndexVisible(this.state.ensureIndexVisible, 'middle')
      }
    })
  }

  getDrillDownData = (routeGroup, regionId, countryId, routeId, flight, type) => {
    var self = this;
    let { gettingDay, gettingYear, gettingMonth, getCabinValue, currency } = this.state;

    apiServices.getRouteProfitabilitySolutionDrillDownData(gettingYear, currency, gettingMonth, gettingDay, routeGroup, regionId, countryId, routeId, flight, getCabinValue, type).then((result) => {
      self.setState({ loading2: false })
      if (result) {
        self.setState({
          drillDownTotalData: result[0].totalData,
          drillDownData: result[0].rowData,
          drillDownColumn: result[0].columnName,
          tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
          regionId: result[0].currentAccess.regionId,
          countryId: result[0].currentAccess.countryId,
          routeId: result[0].currentAccess.routeId,
          flight: result[0].currentAccess.flight,
        })
      }
    });
  }

  getCollapsedMonth = (monthData, gettingMonth, gettingYear) => {
    let month = window.monthNameToNum(gettingMonth)
    let selectedMonthWithDropdown = `► ${window.shortMonthNumToName(month)} ${gettingYear}`
    let selectedMonth = `${window.shortMonthNumToName(month)} ${gettingYear}`
    let monthWithDropDown = Constant.getMonthWithDropdown(monthData).filter((data, index) => {
      if (data.Month.includes('►')) {
        if (selectedMonthWithDropdown === data.Month) {
          data.highlightMonth = true
          this.setState({ ensureIndexVisible: index })
        }
      } else {
        if (selectedMonth === data.Month) {
          data.highlightMonth = true
          this.setState({ ensureIndexVisible: index })
        }
      }
      return data;
    })
    return monthWithDropDown;
  }

  getExpandedMonth = (dayLevelData, gettingDay, gettingMonth, gettingYear) => {
    let insertIndex = null;
    let month = window.monthNameToNum(gettingMonth)
    let selectedMonth = `► ${window.shortMonthNumToName(month)} ${gettingYear}`
    let monthArray = Constant.getMonthWithDropdown(originalMonthData).filter((data, index) => {
      if (selectedMonth === data.Month) {
        insertIndex = index;
        data.Month = `▼ ${(data.Month).substring(2, data.Month.length)}`
        data.highlightMonth = true
        this.setState({ ensureIndexVisible: index })
      }
      return data;
    })

    monthArray.splice(insertIndex + 1, 0, ...dayLevelData)

    if (gettingDay) {
      monthArray = monthArray.filter((data, index) => {
        data.highlightDay = false
        if (gettingDay === data.Month) {
          data.highlightDay = true
          this.setState({ ensureIndexVisible: index })
        }
        return data;
      })
    }
    return monthArray;
  }

  monthWiseCellClick = (params) => {
    var self = this;
    let { routeGroup, regionId, countryId, routeId, flight, type, gettingMonth, gettingYear } = this.state;
    let selectedValue = params.data.Month;
    let daySection = params.data.Day;
    let monthName = params.data.MonthName;
    let monthNum = window.monthNameToNum(monthName);
    let year = parseInt(params.data.Year)
    var column = params.colDef.field;

    this.setState({ gettingMonth: monthName, gettingYear: year, gettingDay: isNaN(parseInt(selectedValue)) ? null : selectedValue })

    if (column === 'Month' && !selectedValue.includes('Total')) {
      if (daySection === 'dayLevel') {
        let RRDateRange = [{
          startDate: `${year}-${monthNum}-${selectedValue}`,
          endDate: `${year}-${monthNum}-${selectedValue}`,
          key: 'selection'
        }]
        window.localStorage.setItem('RRDateRangeValue', JSON.stringify(RRDateRange))
        window.localStorage.setItem('RRDay', 'true')
        let monthData = this.state.monthData.filter((d) => {
          d.highlightDay = false;
          return d;
        })
        params.api.updateRowData({ update: monthData });
      } else {
        window.localStorage.setItem('RRDay', 'false')

        //Collapse and Expand Months
        if (selectedValue.includes('▼') || selectedValue.includes('►')) {
          let RRDateRange = [{
            startDate: Constant.getStartEndDateOfMonth(monthNum, year).startDate,
            endDate: Constant.getStartEndDateOfMonth(monthNum, year).endDate,
            key: 'selection'
          }]
          window.localStorage.setItem('RRDateRangeValue', JSON.stringify(RRDateRange))

          if (gettingMonth === monthName && parseInt(gettingYear) === year) {
            let count = this.state.count;
            this.setState({ count: count + 1 }, () => {
              let _ = Constant.isOdd(this.state.count) ? this.getDayLevelData('monthClick', true) : this.getDayLevelData('monthClick', false)
            })
          } else {
            this.setState({ count: 0 })
            this.getDayLevelData('monthClick', false)
          }
        } //Without Dropdown
        else {
          let RRDateRange = [{
            startDate: Constant.getStartEndDateOfMonth(monthNum, year).startDate,
            endDate: Constant.getStartEndDateOfMonth(monthNum, year).endDate,
            key: 'selection',
            withoutDropDown: true
          }]
          window.localStorage.setItem('RRDateRangeValue', JSON.stringify(RRDateRange))

          let monthData = this.state.monthData.filter((d) => {
            d.highlightMonth = false;
            return d;
          })
          params.api.updateRowData({ update: monthData });
          this.getDayLevelData('monthClick', true)
        }
      }

      if (daySection === 'dayLevel') {
        self.setState({ loading2: true, drillDownData: [], drillDownTotalData: [] })
      }

      self.getDrillDownData(routeGroup, regionId, countryId, routeId, flight, type);

    } else if ((column === '#_VC_Contri') && !selectedValue.includes('Total')) {
      if (daySection !== 'dayLevel') {
        window.localStorage.setItem('RRDay', 'false')
      } else {
        window.localStorage.setItem('RRDay', 'true')
      }
      window.location.pathname = '/routeProfitabilityConsolidate';
    }
  }

  regionCellClick = (params) => {
    var self = this;
    let { routeGroup, regionId, countryId, routeId, flight, getCabinValue } = this.state;
    var column = params.colDef.field;
    var selectedData = `'${params.data.firstColumnName}'`
    var selectedDataWQ = params.data.firstColumnName
    var selectedTitle = params.colDef.headerName

    let found;
    bcData.map((data, i) => data.title === selectedTitle ? found = true : found = false)

    if (column === 'firstColumnName') {
      if (!found) {
        this.storeValuesToLS(regionId, countryId, routeId, flight, getCabinValue, selectedDataWQ);

        if (selectedTitle !== 'Aircraft') {
          self.setState({ selectedData })
          bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
        }
        if (regionId === '*') {
          self.getMonthDrillDownData(routeGroup, selectedData, countryId, routeId, flight)

        } else if (countryId === '*') {
          self.getMonthDrillDownData(routeGroup, regionId, selectedData, routeId, flight)

        } else if (routeId === '*') {
          self.getMonthDrillDownData(routeGroup, regionId, countryId, selectedData, flight)

        } else if (flight === '*') {
          self.getMonthDrillDownData(routeGroup, regionId, countryId, routeId, selectedData)
        }
      }
    }
  }

  rectifyURLValues(regionId, countryId, routeId) {

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

    if (Array.isArray(routeId)) {
      this.selectedRoute = routeId.join(',')
    } else if (regionId.includes("','")) {
      this.selectedRoute = routeId.split("','").join(',')
      this.selectedRoute = this.selectedRoute.substring(1, this.selectedRoute.length - 1);
    } else {
      this.selectedRoute = routeId
      this.selectedRoute = this.selectedRoute.substring(1, this.selectedRoute.length - 1);
    }
  }

  storeValuesToLS(regionId, countryId, routeId, flight, getCabinValue, data) {
    let region = []
    let country = []
    let route = []
    let cabin = []

    this.rectifyURLValues(regionId, countryId, routeId);

    if (regionId === '*') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(data)}`)
      region.push(data)
      window.localStorage.setItem('RouteRegionSelected', JSON.stringify(region))

    } else if (countryId === '*') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${(data)}`)
      country.push(data)
      window.localStorage.setItem('RouteCountrySelected', JSON.stringify(country))

    } else if (routeId === '*') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&Route=${data}`)
      route.push(data)
      window.localStorage.setItem('RouteSelected', JSON.stringify(route))

    } else if (flight === '*') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&Route=${this.selectedRoute}&Flight=${data}`)
      window.localStorage.setItem('RPFlightSelected', data)
    }
  }

  tabClick = (selectedType, outerTab) => {
    var self = this;
    Constant.sendEvent('2', `clicked on ${selectedType} tab`, '/routeProfitabilitySolution', 'Route Profitability Solution');
    let { routeGroup, regionId, countryId, routeId, flight } = this.state;
    self.setState({ type: selectedType, loading2: true, drillDownData: [], drillDownTotalData: [] })

    if (outerTab) {
      this.setState({ outerTab })
    } else {
      this.setState({ outerTab: '' })
    }

    self.getDrillDownData(routeGroup, regionId, countryId, routeId, flight, selectedType)
  }

  homeHandleClick = (e) => {
    var self = this;
    let { routeGroup } = this.state;
    self.setState({ loading: true, loading2: true, firstHome: false, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [], currency: 'bc' })

    window.localStorage.setItem('RouteRegionSelected', 'Null');
    window.localStorage.setItem('RouteCountrySelected', 'Null');
    window.localStorage.setItem('RouteSelected', 'Null');
    window.localStorage.setItem('LegSelected', 'Null');
    window.localStorage.setItem('RPFlightSelected', 'Null');

    self.getMonthDrillDownData(routeGroup, '*', '*', '*', '*', '*')
    bcData = [];
    this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${routeGroup}`)
  }

  listHandleClick = (data, title, selection) => {
    var self = this;
    let { routeGroup, regionId, countryId, routeId, flight } = this.state;
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
        return d.title == title;
      })
      var removeArrayIndex = bcData.slice(0, indexEnd + 1);
      bcData = removeArrayIndex;

      this.changeURLOnListClick(regionId, countryId, routeId, data, getColName)

    } else if (selection === 'browserBack') {
      this.onBackPressClearLS(getColName)
    }

    if (getColName === 'Region') {
      self.getMonthDrillDownData(routeGroup, selectedData, '*', '*', '*', '*')

    } else if (getColName === 'Country') {
      self.getMonthDrillDownData(routeGroup, regionId, selectedData, '*', '*', '*')

    } else if (getColName === 'Route') {
      self.getMonthDrillDownData(routeGroup, regionId, countryId, selectedData, '*', '*')

    } else if (getColName === 'Flight') {
      self.getMonthDrillDownData(routeGroup, regionId, countryId, routeId, selectedData, '*')

    }
    //  else if (getColName === 'Flight') {
    //   self.getMonthDrillDownData(routeGroup, regionId, countryId, routeId, selectedData)

    // } 
    else if (getColName === 'RouteGroup') {
      self.setState({ routeGroup: data }, () => this.homeHandleClick())
    }

  }

  changeURLOnListClick(regionId, countryId, routeId, selectedData, getColName) {

    this.rectifyURLValues(regionId, countryId, routeId);

    if (getColName === 'Region') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(selectedData)}`)
      window.localStorage.setItem('RouteCountrySelected', 'Null');
      window.localStorage.setItem('RouteSelected', 'Null');
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    } else if (getColName === 'Country') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${(selectedData)}`)
      window.localStorage.setItem('RouteSelected', 'Null');
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    } else if (getColName === 'Route') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&Route=${selectedData}`)
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    } else if (getColName === 'Flight') {
      this.props.history.push(`/routeProfitabilitySolution?RouteGroup=${this.state.routeGroup}&Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&Route=${this.selectedRoute}&Flight=${selectedData}`)
    }
  }

  onBackPressClearLS(getColName) {

    if (getColName === 'Region') {
      window.localStorage.setItem('RouteCountrySelected', 'Null');
      window.localStorage.setItem('RouteSelected', 'Null');
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    } else if (getColName === 'Country') {
      window.localStorage.setItem('RouteSelected', 'Null');
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    } else if (getColName === 'Route') {
      window.localStorage.setItem('LegSelected', 'Null');
      window.localStorage.setItem('RPFlightSelected', 'Null');

    }
  }

  closeChartModal() {
    this.setState({ chartVisible: false })
  }

  closeTableModal() {
    this.setState({ tableModalVisible: false })
  }

  callAccess(routeGroup) {
    let routeGroupArray = [];
    routeGroupArray.push(routeGroup)
    window.localStorage.setItem('RouteGroupSelected', JSON.stringify(routeGroupArray));
    this.setState({ routeGroup }, () => this.homeHandleClick())
  }

  showLoader = () => {
    $("#loaderImage").addClass("loader-visible")
  }

  hideLoader = () => {
    $("#loaderImage").removeClass("loader-visible")
    $(".x_panel").addClass("opacity-fade");
    $(".top-buttons").addClass("opacity-fade");
  }

  renderTabs() {
    let { tabName, gettingMonth, regionId, countryId, routeId, flight, getCabinValue, routeGroup, gettingYear, type, routeParam, aircraftParam, outerTab } = this.state;
    //const downloadURLDrillDown = apiServices.exportCSVRouteProfitabilitySolutionDrillDownURL(gettingYear, gettingMonth, routeGroup, regionId, countryId, routeId, flight, getCabinValue, type)
    //const downloadURLMonthly = apiServices.exportCSVRouteProfitabilitySolutionMonthlyURL(routeGroup, regionId, countryId, routeId, flight, getCabinValue)
    const downloadURLDrillDown = localStorage.getItem('RouteProfitabilityDrillDownDownloadURL')
    const downloadURLMonthly = localStorage.getItem('RouteProfitabilityMonthlyDownloadURL')

    return (
      <ul className="nav nav-tabs" role="tablist">

        {tabName === 'Aircraft' || tabName === 'Flight' ?
          <li role="presentation" className={`${routeParam ? 'active' : ""}`} onClick={() => this.tabClick('OD', 'Route')}>
            <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
              Route
            </a>
          </li> : ""}

        {tabName === 'Aircraft' ?
          <li role="presentation" onClick={() => this.tabClick('Flights', 'Flight')}>
            <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
              Flight
            </a>
          </li> : ''}

        <li role="presentation" className={`${aircraftParam ? '' : routeParam ? '' : "active"}`} onClick={() => this.tabClick('Null')}>
          <a href="#Section1" aria-controls="profile" role="tab" data-toggle="tab">
            {tabName}
          </a>
        </li>

        {tabName === 'Route' ? '' : tabName === 'Aircraft' ? '' : tabName === 'Flight' ? '' :
          <li role="presentation" className={`${routeParam ? 'active' : outerTab === 'Route' ? "active" : ""}`} onClick={() => this.tabClick('OD')}>
            <a href="#Section2" aria-controls="messages" role="tab" data-toggle="tab">
              Route
            </a>
          </li>}


        {tabName === 'Flight' ? '' : tabName === 'Aircraft' ? '' :
          <li role="presentation" className={`${outerTab === 'Flight' ? "active" : ""}`} onClick={() => this.tabClick('Flights')}>
            <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
              Flight
            </a>
          </li>}

        {tabName === 'Aircraft' ? '' :
          <li id='regionTab' role="presentation" className={`${aircraftParam ? 'active' : ""}`} onClick={() => this.tabClick('Aircraft')}>
            <a href="#Section4" aria-controls="profile" role="tab" data-toggle="tab">
              Aircraft
          </a>
          </li>}

        {/* <li id='regionTab' role="presentation" onClick={this.regionTabClick}>
          <a href="#Section5" aria-controls="profile" role="tab" data-toggle="tab">
            Flight based
          </a>
        </li> */}

        <div className='RRPBtns'>
        <DownloadCSV url={downloadURLDrillDown} name={`DRILLDOWN DATA`} path={`/routeProfitabilitySolution`} page={`Route Profitability Solution Page`} />
        <DownloadCSV url={downloadURLMonthly} name={`MONTHLY DATA`} path={`/routeProfitabilitySolution`} page={`Route Profitability Solution Page`} />
        {/* {routeId !== '*' ? <button className='btn download' onClick={this.posContributionClick}>POS Contribution</button> : ''} */}
        </div>
      </ul>
    )
  }

  gridApiMonthly = (api) => {
    this.gridApiMonth = api;
  }

  render() {
    const { aircraftParam, routeParam, cabinDisable, routeId, type, routeGroup, accessLevelDisable, firstLoadList, regionLevelAccess } = this.state;
    window.localStorage.setItem('RRBCData', JSON.stringify(bcData))
    return (
      <div className='routeProfitabilitySolution'>
        <TopMenuBar dashboardPath={'/routeProfitability'} {...this.props} />
        <Loader />
        <div className="row">
          <div className="col-md-12 col-sm-12 col-xs-12 top">
            <div className="navdesign" style={{ marginTop: '0px' }}>
              <div className="col-md-7 col-sm-7 col-xs-12 toggle1">
                <h2>Route Profitability</h2>
                {firstLoadList ? "" :
                  <div className='route-access'>
                    {routeGroup}
                    <div className="triangle-up"></div>
                    <div className='route-groups'>
                      <div className={`route-main ${accessLevelDisable ? ' route-main-disable' : ''}`}>
                        <span className={`${accessLevelDisable ? ' route-access-disable' : ''}`} onClick={() => this.callAccess('Network')}>
                          Network
                          </span>
                      </div>
                      <div className={`route-main ${accessLevelDisable ? regionLevelAccess ? 'route-main-disable' : routeGroup === 'Domestic' ? '' : 'route-main-disable' : ''}`}>
                        <span className={`${accessLevelDisable ? regionLevelAccess ? 'route-access-disable' : routeGroup === 'Domestic' ? '' : 'route-access-disable' : ''}`} onClick={() => this.callAccess('Domestic')}>
                          Domestic
                          </span>
                      </div>
                      <div className={`route-main international ${accessLevelDisable ? regionLevelAccess ? 'route-main-disable' : routeGroup === 'International' ? '' : 'route-main-disable' : ''}`}>
                        <span className={`${accessLevelDisable ? regionLevelAccess ? 'route-access-disable' : routeGroup === 'International' ? '' : 'route-access-disable' : ''}`} onClick={() => this.callAccess('International')}>
                          International
                          </span>
                      </div>
                    </div>
                  </div>
                  //  <select className="form-control cabinselect currency-dropdown route-access" onChange={(e) => this.callAccess(e)} disabled={this.state.accessLevelDisable ? true : false}>
                  //   <option value='Network'>Network</option>
                  //   <option value='Domestic' selected={this.state.routeGroup === 'Domestic'}>Domestic</option>
                  //   <option value='International' selected={this.state.routeGroup === 'International'}>International</option>
                  // </select>
                }
                <section>
                  <nav>
                    <ol className="cd-breadcrumb">
                      {/* {this.state.baseAccess !== '' ? <li onClick={this.homeHandleClick} > {this.state.baseAccess} </li> : ''} */}
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

              <div className="col-md-5 col-sm-5 col-xs-12 toggle2">
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
                {/* <div className='cabin-selection'>
                  <h4>Aircraft Type :</h4>
                  <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.aircraft(e)} >
                    <option value='all'>All</option>
                    <option value='doc'>DOC</option>
                  <option value='tc'>TC</option>
                  </select>
                </div>
                <h4>Cost Type :</h4>
                <select className="form-control cabinselect currency-dropdown" onChange={(e) => this.currency(e)} >
                  <option value='vc' selected={this.state.countryId === '*' || this.state.toggle === 'vc' ? true : false}>VC</option>
                  <option value='doc'>DOC</option>
                  <option value='tc'>TC</option>
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
                  rowData={this.state.monthData}
                  columnDefs={this.state.routeMonthColumns}
                  onCellClicked={(cellData) => this.monthWiseCellClick(cellData)}
                  loading={this.state.loading}
                  rowClassRules={this.state.routeMonthRowClassRule}
                  frameworkComponents={{ customHeaderGroupComponent: RouteCustomHeaderGroup }}
                  route={true}
                  ensureIndexVisible={this.state.ensureIndexVisible}
                />
                <TotalRow
                  rowData={this.state.monthTotalData}
                  columnDefs={this.state.routeMonthColumns}
                  loading={this.state.loading}
                  frameworkComponents={{ customHeaderGroupComponent: RouteCustomHeaderGroup }}
                  responsive={true}
                  reducingPadding={true}
                />

                <div className="tab" id="posTableTab" role="tabpanel" style={{ marginTop: '10px' }}>

                  {this.renderTabs()}

                  <div className="tab-content tabs">

                    <div role="tabpanel" className={`tab-pane fade in ${aircraftParam ? '' : routeParam ? '' : "active"}`} id="Section1">

                      {/* Region */}
                      <DataTableComponent
                        rowData={this.state.drillDownData}
                        columnDefs={this.state.drillDownColumn}
                        onCellClicked={(cellData) => this.regionCellClick(cellData)}
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
                    <div role="tabpanel" className={`tab-pane fade in ${routeParam ? 'active' : ""}`} id="Section2">


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

                    {/* Flight */}
                    <div role="tabpanel" className="tab-pane fade in" id="Section3">

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

                    {/* Aircraft */}
                    <div role="tabpanel" className={`tab-pane fade in ${aircraftParam ? 'active' : ""}`} id="Section4">

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
        </div>

      </div >

    );
  }
}

const NewComponentRoute = BrowserToProps(RouteProfitabilitySolution);

export default NewComponentRoute;

