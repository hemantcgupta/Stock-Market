import $ from 'jquery';
import React, { Component } from 'react';
import BrowserToProps from 'react-browser-to-props';
import APIServices from '../../API/apiservices';
import eventApi from '../../API/eventApi';
import '../../App';
import AlertModal from '../../Component/AlertModal';
import ChartModelDetails from '../../Component/chartModel';
import DataTableComponent from '../../Component/DataTableComponent';
import DatatableModelDetails from '../../Component/dataTableModel';
import Loader from '../../Component/Loader';
import TopMenuBar from '../../Component/TopMenuBar';
import TotalRow from '../../Component/TotalRow';
import cookieStorage from '../../Constants/cookie-storage';
import { default as Constant, default as String } from '../../Constants/validator';
import '.././PosDetails//PosDetails.scss';
import POSCustomHeaderGroup from '../PosDetails/POSCustomHeaderGroup';
import "./index.scss";


const apiServices = new APIServices();
const currentYear = new Date().getFullYear()

let originalMonthData = [];
let originalDrillDownData = []

let bcData = [];

class AlertsNew extends Component {
    constructor(props) {
        super(props);
        this.pathName = window.location.pathname;
        this.selectedRegion = null;
        this.selectedCountry = null;
        this.selectedCity = null;
        this.gridApiMonth = null;
        this.gridApiDrillDown = null;
        this.directPOS = false;
        this.state = {
            monthcolumns: [],
            monthData: [],
            monthTotalData: [],
            drillDownColumn: [],
            drillDownData: [],
            drillDownTotalData: [],
            alertTrendColumn: [],
            alertTrendData: [],
            alertTrendTotalData: [],
            alertTrendType: null,
            trendType: null,
            gettingMonth: null,
            gettingYear: null,
            cabinOption: [],
            alertData: [],
            getCabinValue: 'Null',
            cabinSelectedDropDown: 'Null',
            cabinDisable: true,
            toggle: 'bc',
            alertVisible: false,
            tableModalVisible: false,
            tabName: 'Region',
            regionId: '*',
            countryId: '*',
            cityId: '*',
            route: '*',
            type: 'Alert',
            priority: 'Null',
            selectedData: 'Null',
            loading: false,
            loading2: false,
            loading3: false,
            firstLoadList: false,
            accessLevelDisable: false,
            posMonthRowClassRule: {
                'highlight-row': 'data.highlightMe'
            },
            firstHome: true,
            ensureIndexVisible: null,
            ensureIndexVisibleDD: null,
            interline: true,
            count: 0,
            drillDownCount: 0,
            gettingAction: null,
            alertType: null,
            chartVisible: false,
            chartHeader: "",
            selectedDataChart: null
        }
        this.sendEvent('1', 'viewed Alert Page', 'alert', 'Alert Page');
    }

    sendEvent = (id, description, path, page_name) => {
        const eventData = {
            event_id: `${id}`,
            description: `User ${description}`,
            where_path: `/${path}`,
            page_name: `${page_name}`
        }
        eventApi.sendEvent(eventData)
    }

    componentDidMount() {
        const self = this;
        self.getFiltersValue()

        apiServices.getClassNameDetails().then((result) => {
            if (result) {
                const classData = result[0].classDatas;
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
        }
        if (obj.hasOwnProperty('O%26D') && !bcData.some(function (o) { return o["title"] === "O&D"; })) {

            bcData.push({ "val": obj['O%26D'], "title": 'O&D' });
            console.log(obj);
            self.setState({ route: obj['O%26D'] })
            window.localStorage.setItem('ODSelected', obj['O%26D'])
        }

        if (bcData.length > 0) {
            const removeArrayIndex = bcData.slice(0, lastIndex + 1);
            bcData = removeArrayIndex;
        }

        this.listHandleClick(data[lastIndex], title[lastIndex], 'browserBack')
    }

    getFiltersValue = () => {
        bcData = []
        let RegionSelected = window.localStorage.getItem('RegionSelected')
        let CountrySelected = window.localStorage.getItem('CountrySelected')
        let CitySelected = window.localStorage.getItem('CitySelected')
        let ODSelected = window.localStorage.getItem('ODSelected')
        let rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'))

        CitySelected = CitySelected === null || CitySelected === 'Null' || CitySelected === '' ? '*' : JSON.parse(CitySelected)

        this.setState({
            regionId: RegionSelected === null || RegionSelected === 'Null' || RegionSelected === '' ? '*' : JSON.parse(RegionSelected),
            countryId: CountrySelected === null || CountrySelected === 'Null' || CountrySelected === '' ? '*' : JSON.parse(CountrySelected),
            cityId: CitySelected,
            route: ODSelected === null || ODSelected === 'Null' || ODSelected === '' || ODSelected === '*' ? '*' : JSON.parse(ODSelected),
            gettingMonth: window.monthNumToName(rangeValue.from.month),
            gettingYear: rangeValue.from.year,
            getCabinValue: 'Null',
            cabinSelectedDropDown: 'Null'
        }, () => this.getInitialData())
    }

    getInitialData = () => {
        const self = this;
        let { gettingMonth, gettingYear, regionId, countryId, cityId, route, getCabinValue } = this.state;
        self.setState({ loading: true, loading2: true, firstLoadList: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })

        self.getInitialListData(regionId, countryId, cityId, route);

        apiServices.getAlertMonthTables("Network", regionId, countryId, cityId, route, getCabinValue).then(function (result) {
            self.setState({ loading: false, firstLoadList: false })
            let isCollapseMonth = Constant.isOdd(self.state.count);
            if (result) {
                originalMonthData = result[0].rowData;
                const totalData = result[0].totalData;
                const columnName = result[0].columnName;
                const alertMonthdata = result[0].rowData;

                gettingMonth = result[0].rowData[0].MonthName

                const range = { from: { year: gettingYear, month: window.monthNameToNum(gettingMonth) }, to: { year: gettingYear, month: window.monthNameToNum(gettingMonth) } }
                window.localStorage.setItem('rangeValue', JSON.stringify(range))

                self.setState({
                    monthData: self.getHighlightedMonth(alertMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData, gettingMonth
                }, () => self.getMonthAlertDrillDownData(regionId, countryId, cityId, route, '', isCollapseMonth))
            }

            self.getDrillDownData(regionId, countryId, cityId, route, 'Alert');
        });
    }

    getInitialListData = (regionId, countryId, cityId, OD) => {
        const self = this;
        const userDetails = JSON.parse(cookieStorage.getCookie('userDetails'));
        // let route = OD.substring(1, OD.length - 1)
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
            bcData.push({ "val": regionId, "title": 'Region', 'disable': country !== '*' });
            self.setState({ selectedData: regionId })
        }
        if (countryId !== '*') {
            bcData.push({ "val": countryId, "title": 'Country', 'disable': city !== '*' });
            self.setState({ selectedData: countryId })
        }
        if (cityId !== '*') {
            bcData.push({ "val": cityId, "title": 'POS' });
            self.setState({ selectedData: cityId })
        }
        if (cityId !== '*') {
            if (OD !== '*') {
                self.setState({ selectedData: OD })
            }
        }
    }

    getMonthDrillDownData = (regionId, countryId, cityId, route, cabin = "Null") => {
        const self = this;
        let { gettingMonth, type, gettingYear, toggle, priority } = this.state;

        self.setState({ loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
        let isCollapseMonth = Constant.isOdd(this.state.count);
        apiServices.getAlertMonthTables(toggle, regionId, countryId, cityId, route, cabin).then(function (result) {
            self.setState({ loading: false })
            if (result) {
                originalMonthData = result[0].rowData;
                const totalData = result[0].totalData
                const columnName = result[0].columnName;
                const alertMonthdata = result[0].rowData;
                self.setState({ monthData: self.getHighlightedMonth(alertMonthdata, gettingMonth, gettingYear), monthcolumns: columnName, monthTotalData: totalData },
                    () => self.getMonthAlertDrillDownData(regionId, countryId, cityId, route, '', isCollapseMonth))
            }
        });

        apiServices.getAlertDrillDownData(gettingYear, gettingMonth, "Network", regionId, countryId, cityId, route, cabin, type, priority).then((result) => {
            self.setState({ loading2: false })
            let isCollapseAction = Constant.isOdd(this.state.drillDownCount);
            if (result) {
                originalDrillDownData = result[0].rowData
                self.setState({
                    drillDownTotalData: result[0].totalData,
                    drillDownData: result[0].rowData,
                    drillDownColumn: result[0].columnName,
                    tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
                    regionId: result[0].currentAccess.regionId,
                    countryId: result[0].currentAccess.countryId,
                    cityId: result[0].currentAccess.cityId,
                    route: result[0].currentAccess.routeId,
                }, () => self.getActionDrillDownData(regionId, countryId, cityId, route, '', isCollapseAction))
            }
        });
    }

    getMonthAlertDrillDownData(regionId, countryId, cityId, route, click, isCollapseMonth, alertType) {
        let { routeGroup, gettingMonth, gettingYear, getCabinValue } = this.state;
        if (isCollapseMonth) {
            this.setState({
                monthData: this.getCollapsedMonth(originalMonthData, gettingMonth, gettingYear)
            })
        } else {
            if (click === 'monthClick') this.showLoader();
            apiServices.getActionDrillDown(gettingYear, gettingMonth, routeGroup, regionId, countryId, cityId, route, getCabinValue, "Null", "Monthly", alertType).then((result) => {
                this.setState({ loading: false })
                this.hideLoader()
                if (result) {
                    const alertDrillDown = result[0].rowData;
                    this.setState({
                        monthData: this.getExpandedMonth(alertDrillDown, gettingMonth, gettingYear)
                    })
                    // this.gridApiMonth.ensureIndexVisible(this.state.ensureIndexVisible, 'middle')
                }
            })
        }
    }

    getActionDrillDownData(regionId, countryId, cityId, route, click, isCollapseAction) {
        if (this.state.type !== "Alert") return
        let { routeGroup, gettingMonth, gettingYear, getCabinValue, gettingAction } = this.state;
        this.setState({ drillDownData: [] })
        if (isCollapseAction) {
            this.setState({
                drillDownData: this.getCollapsedAction(originalDrillDownData, gettingAction)
            })
        } else {
            if (click === 'monthClick') this.showLoader()
            apiServices.getActionDrillDown(gettingYear, gettingMonth, routeGroup, regionId, countryId, cityId, route, getCabinValue, gettingAction, "DrillDown").then((result) => {
                this.setState({ loading: false })
                this.hideLoader()
                if (result) {
                    const actionDrillDown = result[0].rowData;
                    this.setState({
                        drillDownData: isCollapseAction ? this.getCollapsedAction(originalDrillDownData, gettingAction) : this.getExpandedAction(actionDrillDown, gettingAction)
                    })
                    this.gridApiDrillDown.ensureIndexVisible(this.state.ensureIndexVisibleDD, 'top')
                }
            }).catch((e) => {
                this.setState({ loading: false })
                this.hideLoader()
                console.log(e)
            })
        }
    }

    getDrillDownData = (regionId, countryId, cityId, route, type) => {
        const self = this;
        let { gettingYear, gettingMonth, getCabinValue, priority } = this.state;

        apiServices.getAlertDrillDownData(gettingYear, gettingMonth, "Network", regionId, countryId, cityId, route, getCabinValue, type, priority).then((result) => {
            self.setState({ loading2: false })
            let isCollapseAction = Constant.isOdd(self.state.drillDownCount);
            if (result) {
                originalDrillDownData = result[0].rowData
                self.setState({
                    drillDownTotalData: result[0].totalData,
                    drillDownData: result[0].rowData,
                    drillDownColumn: result[0].columnName,
                    tabName: type === 'Null' ? result[0].tabName : result[0].firstTabName,
                    regionId: result[0].currentAccess.regionId,
                    countryId: result[0].currentAccess.countryId,
                    cityId: result[0].currentAccess.cityId,
                    route: result[0].currentAccess.routeId
                }, () => self.getActionDrillDownData(regionId, countryId, cityId, route, '', isCollapseAction))
            }
        });
    }

    getActionDataDrillDown = (regionId, countryId, cityId, route, getCabinValue, alertType, action, priority) => {
        const self = this;
        let { gettingYear, gettingMonth } = this.state;

        this.directPOS = true
        apiServices.getActionDataDrillDown(gettingYear, gettingMonth, "Network", regionId, countryId, cityId, route, getCabinValue, alertType, action, priority).then((result) => {
            self.setState({ loading2: false })
            if (result) {
                originalDrillDownData = result[0].rowData
                self.setState({
                    drillDownTotalData: result[0].totalData,
                    drillDownData: result[0].rowData,
                    drillDownColumn: result[0].columnName,
                    regionId: result[0].currentAccess.regionId,
                    countryId: result[0].currentAccess.countryId,
                    cityId: result[0].currentAccess.cityId,
                    route: result[0].currentAccess.routeId,
                    tabName: result[0].tabName
                })
            }
        });
        let isCollapseMonth = Constant.isOdd(this.state.count);

        apiServices.getAlertMonthTables("Network", regionId, countryId, cityId, route, getCabinValue, alertType).then((result) => {
            self.setState({ loading: false })
            if (result) {
                originalMonthData = result[0].rowData;
                self.getMonthAlertDrillDownData(regionId, countryId, cityId, route, '', isCollapseMonth, alertType)
            }
        })
    }

    getHighlightedMonth(alertMonthdata, month, year) {
        let monthNumber = window.monthNameToNum(month)
        let count = 0;
        let data = alertMonthdata.forEach((data, index) => {
            const monthName = data.Month;
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

    getCollapsedMonth = (monthData, gettingMonth, gettingYear) => {
        let month = window.monthNameToNum(gettingMonth)
        let selectedMonthWithDropdown = `► ${window.shortMonthNumToName(month)} ${gettingYear}`
        let selectedMonth = `${window.shortMonthNumToName(month)} ${gettingYear}`
        let monthWithDropDown = Constant.getAlertMonthWithDropdown(monthData).filter((data, index) => {
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

    getExpandedMonth = (alertData, gettingMonth, gettingYear) => {
        let insertIndex = null;
        let month = window.monthNameToNum(gettingMonth)
        let selectedMonth = `► ${window.shortMonthNumToName(month)} ${gettingYear}`
        let monthArray = Constant.getAlertMonthWithDropdown(originalMonthData).filter((data, index) => {
            if (selectedMonth === data.Month) {
                insertIndex = index;
                data.Month = `▼ ${(data.Month).substring(2, data.Month.length)}`
                data.highlightMonth = true
                this.setState({ ensureIndexVisible: index })
            }
            return data;
        })

        monthArray.splice(insertIndex + 1, 0, ...alertData)
        return monthArray;
    }

    getCollapsedAction = (drilldownData, action) => {
        let selectedActionWithDropdown = `► ${action}`
        let selectedAction = action
        let actionWithDropDown = Constant.getActionWithDropdown(drilldownData).filter((data, index) => {
            if (data.firstColumnName.includes('►')) {
                if (selectedActionWithDropdown === data.firstColumnName) {
                    data.highlightMonth = true
                    this.setState({ ensureIndexVisibleDD: index })
                }
            } else {
                if (selectedAction === data.firstColumnName) {
                    data.highlightMonth = true
                    this.setState({ ensureIndexVisibleDD: index })
                }
            }
            return data;
        })
        return actionWithDropDown;
    }

    getExpandedAction = (alertData, action) => {
        let insertIndex = null;
        let selectedAction = `► ${action}`
        let drilldownArray = Constant.getActionWithDropdown(originalDrillDownData).filter((data, index) => {
            if (selectedAction === data.firstColumnName) {
                insertIndex = index;
                data.firstColumnName = `▼ ${(data.firstColumnName).substring(2, data.firstColumnName.length)}`
                data.highlightMonth = true
                this.setState({ ensureIndexVisibleDD: index })
            }
            return data;
        })

        drilldownArray.splice(insertIndex + 1, 0, ...alertData)
        return drilldownArray;
    }

    monthWiseCellClick = (params) => {
        const monththis = this;
        monththis.sendEvent('2', 'clicked on Months row', 'alert', 'Alert Page');
        let { gettingMonth, regionId, countryId, cityId, route, getCabinValue, type, monthcolumns } = this.state;
        let selectedMonth = params.data.Month;
        let monthName = params.data.MonthName;
        const column = params.colDef.field;

        const monthData = this.state.monthData.map((d) => {
            d.highlightMe = false;
            return d;
        })
        params.api.updateRowData({ update: monthData });

        if (!params.data.isAlert) {
            monththis.setState({ gettingMonth: params.data.MonthName, gettingYear: params.data.Year })
            const range = { from: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) }, to: { year: params.data.Year, month: window.monthNameToNum(params.data.MonthName) } }
            window.localStorage.setItem('rangeValue', JSON.stringify(range))
            this.gridApiMonth.setColumnDefs(monthcolumns);
        }

        if ((column === "AC_AL" || column === "PN_AL" || column === "RE_AL" || column === "RJ_AL" || column === "TL_AL") && !selectedMonth.includes('Total')) {
            params.event.stopPropagation();
            const self = this;
            let { regionId, countryId, cityId, route, monthcolumns } = this.state;
            window.$('.nav-tabs li a[href="#Section1"]').tab('show')

            self.setState({ drillDownData: [], drillDownTotalData: [], loading2: true })
            const alertType = column === "AC_AL" ? "Alert_Action" :
                column === "PN_AL" ? "Alert_Pending" :
                    column === "TL_AL" ? "Total" :
                        column === "RE_AL" ? "Reccuring" :
                            "Rejected"

            self.setState({ alertType })
            if (params.data.isAlert) {
                self.setState({ priority: params.data.Month })
                self.getActionDataDrillDown(regionId, countryId, cityId, route, getCabinValue, alertType, undefined, params.data.Month)
            } else {
                self.getActionDataDrillDown(regionId, countryId, cityId, route, getCabinValue, alertType)
            }

            this.gridApiMonth.setColumnDefs(monthcolumns);
        }
        else if (column === "CY_B" && !selectedMonth.includes('Total') && !params.data.isAlert) {
            params.event.stopPropagation();
            monththis.showLoader();
            apiServices.getBookingTable(params.data.Year, params.data.MonthName, regionId, countryId, cityId, route, getCabinValue).then((result) => {
                monththis.hideLoader();
                if (result) {
                    const columnName = result[0].columnName;
                    const rowData = result[0].cabinData;
                    const totalData = result[0].totalData
                    monththis.setState({ tableModalVisible: true, alertTrendData: rowData, alertTrendColumn: columnName, alertTrendType: "Booking", alertTrendTotalData: totalData, interline: false })
                }
            });
        }
        else if (column === "AL_F" && !selectedMonth.includes('Total') && !params.data.isAlert) {
            monththis.setState({ chartVisible: true, chartHeader: "Price", selectedDataChart: null })
        }
        else if ((column === "CY_R" || column === "CY_P" || column === "CY_A" || column === "AL_MS" || column === "AV_S") && !selectedMonth.includes('Total') && !params.data.isAlert) {
            params.event.stopPropagation();
            monththis.showLoader();
            const trendType = column === "CY_R" ? "Revenue" :
                column === "CY_P" ? "Passenger" :
                    column === "CY_A" ? "AvgFare" :
                        column === "AL_MS" ? "MarketShare" :
                            "Avail"
            apiServices.getAlertTrendData(params.data.Year, params.data.MonthName, regionId, countryId, cityId, route, getCabinValue, trendType).then((result) => {
                monththis.hideLoader();
                if (result) {
                    const columnName = result[0].columnName;
                    const rowData = result[0].rowData;
                    monththis.setState({ tableModalVisible: true, alertTrendData: rowData, alertTrendColumn: columnName, alertTrendType: result[0].name, alertTrendTotalData: [], interline: true })
                }
            }).catch((e) => {
                monththis.hideLoader();
            });

        } else if (column === 'Month' && !selectedMonth.includes('Total')) {
            //Collapse and Expand Months
            if (selectedMonth.includes('▼') || selectedMonth.includes('►')) {
                if (gettingMonth === monthName) {
                    let count = this.state.count;
                    this.setState({ count: count + 1 }, () => {
                        this.getMonthAlertDrillDownData(regionId, countryId, cityId, route, 'monthClick', Constant.isOdd(this.state.count))
                    })
                } else {
                    this.setState({ count: 0 })
                    this.getMonthAlertDrillDownData(regionId, countryId, cityId, route, 'monthClick', false)
                }
                monththis.setState({ loading2: true, drillDownData: [], drillDownTotalData: [], priority: "Null" })
                monththis.getDrillDownData(regionId, countryId, cityId, route, type)
            }
            else if (params.data.isAlert) {
                params.event.stopPropagation();
                const priority = params.data.Month
                monththis.setState({ priority }, () => {
                    this.getDrillDownData(regionId, countryId, cityId, route, type)
                })
                //     let { regionId, countryId, cityId, route, monthcolumns } = this.state;
                //     if (cityId === "*") {
                //         window.$('.nav-tabs li a[href="#Section4"]').tab('show')
                //     } else {
                //         window.$('.nav-tabs li a[href="#Section1"]').tab('show')
                //         self.setState({ tabName: "OD" })
                //     }
                //     self.setState({ drillDownData: [], drillDownTotalData: [], loading2: true })
                //     self.getActionDataDrillDown(regionId, countryId, cityId, route, "TL_AL")
                //     this.gridApiMonth.setColumnDefs(monthcolumns);
            }
            //Without Dropdown
            else {
                // let monthData = this.state.monthData.filter((d) => {
                //     d.highlightMonth = false;
                //     return d;
                // })
                //     params.api.updateRowData({ update: monthData });
                //     this.getMonthAlertDrillDownData('monthClick', true)
            }
        }
    }

    regionCellClick = (params) => {
        const self = this;
        self.sendEvent('2', 'clicked on Region drill down', 'alert', 'Alert Page');
        let { regionId, countryId, cityId, route, alertType, priority, getCabinValue, gettingAction } = this.state;

        const column = params.colDef.field;
        const selectedData = `'${params.data.firstColumnName}'`;
        const selectedDataWQ = params.data.firstColumnName;
        const selectedTitle = params.colDef.headerName
        let action = params.data.actionName;
        const hyperLink = params.colDef.underline
        let found;
        bcData.forEach((data) => found = data.title === selectedTitle)

        console.log("Alert::", params.data.isAlert, this.directPOS);

        // Actions will have the dropdown arrow at start
        if (selectedDataWQ.includes('▼') || selectedDataWQ.includes('►')) {
            self.setState({ gettingAction: params.data.actionName })
        }

        if (column === 'firstColumnName') {
            if (!found) {
                if (selectedTitle === "Action") {
                    if (selectedDataWQ.includes('▼') || selectedDataWQ.includes('►')) {
                        if (gettingAction === action) {
                            let drillDownCount = this.state.drillDownCount;
                            this.setState({ drillDownCount: drillDownCount + 1 }, () => {
                                this.getActionDrillDownData(regionId, countryId, cityId, route, 'monthClick', Constant.isOdd(this.state.drillDownCount))
                            })
                        } else {
                            this.setState({ drillDownCount: 0 })
                            this.getActionDrillDownData(regionId, countryId, cityId, route, 'monthClick', false)
                        }
                    }
                    //Without Dropdown
                    else {
                        const self = this;
                        let { regionId, countryId, cityId, route, gettingAction } = this.state;
                        window.$('.nav-tabs li a[href="#Section1"]').tab('show')
                        self.setState({ tabName: "POS" })
                        self.setState({ drillDownData: [], drillDownTotalData: [], loading2: true, priority: params.data.firstColumnName })
                        self.getActionDataDrillDown(regionId, countryId, cityId, route, getCabinValue, undefined, gettingAction, params.data.firstColumnName)
                        this.directPOS = true
                        let monthData = this.state.drillDownData.filter((d) => {
                            d.highlightMonth = false;
                            return d;
                        })
                        params.api.updateRowData({ update: monthData });
                    }

                    return
                }
                if (!hyperLink) return

                if (selectedTitle === "Cabin") {
                    window.localStorage.setItem('SelectedCabin', JSON.stringify([selectedDataWQ]))
                    this.props.history.push('/alertDetails')
                    return
                }

                if (this.directPOS) {
                    this.storeValuesToLS(regionId, countryId, cityId, route, getCabinValue, selectedDataWQ);
                    if (cityId === '*') {
                        this.getActionDataDrillDown(regionId, countryId, selectedData, route, getCabinValue, alertType, gettingAction, priority)
                        this.setState({ tabName: "OD" })
                    } else if (route === '*') {
                        this.getActionDataDrillDown(regionId, countryId, cityId, selectedData, getCabinValue, alertType, gettingAction, priority)
                        this.setState({ tabName: "Cabin" })
                    }
                    self.setState({ selectedData })
                    bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
                    return
                }

                this.storeValuesToLS(regionId, countryId, cityId, route, getCabinValue, selectedDataWQ);

                if (selectedTitle !== 'Cabin') {
                    self.setState({ selectedData })
                    bcData.push({ "val": selectedDataWQ, "title": selectedTitle })
                }

                if (regionId === '*') {
                    self.getMonthDrillDownData(selectedData, countryId, cityId, route)
                } else if (countryId === '*') {
                    self.getMonthDrillDownData(regionId, selectedData, cityId, route)
                } else if (cityId === '*') {
                    self.getMonthDrillDownData(regionId, countryId, selectedData, route)
                } else if (route === '*') {
                    self.getMonthDrillDownData(regionId, countryId, cityId, selectedData)
                } else if (getCabinValue === 'Null') {
                    self.getMonthDrillDownData(regionId, countryId, cityId, route, selectedData)
                }
            }
            console.log("Column", column);
        } else if (column === "AL_F") {
            console.log("Data", selectedData, selectedDataWQ);
            self.setState({ chartVisible: true, chartHeader: "Price", selectedDataChart: selectedData })
            return
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

    storeValuesToLS(regionId, countryId, cityId, route, getCabinValue, data) {
        let region = []
        let country = []
        let city = []
        let od = []
        let cabin = []
        const directPOS = this.directPOS

        this.rectifyURLValues(regionId, countryId, cityId);

        if (regionId === '*' && !directPOS) {
            this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(data)}`)
            region.push(data)
            window.localStorage.setItem('RegionSelected', JSON.stringify(region))

        } else if (countryId === '*' && !directPOS) {
            this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${(data)}`)
            country.push(data)
            window.localStorage.setItem('CountrySelected', JSON.stringify(country))

        } else if (cityId === '*') {
            if (directPOS) {
                this.props.history.push(`${this.pathName}?POS=${data}`)
            } else {
                this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${data}`)
            }
            city.push(data)
            window.localStorage.setItem('CitySelected', JSON.stringify(city))

        } else if (route === '*') {
            if (directPOS) {
                this.props.history.push(`${this.pathName}?POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${data}`)
            } else {
                this.props.history.push(`${this.pathName}?Region=${encodeURIComponent(this.selectedRegion)}&Country=${this.selectedCountry}&POS=${this.selectedCity}&${encodeURIComponent('O&D')}=${data}`)
            }
            od.push(data)
            window.localStorage.setItem('ODSelected', JSON.stringify(od))
        } else if (getCabinValue === 'Null') {
            cabin.push(data)
            window.localStorage.setItem('SelectedCabin', JSON.stringify(cabin))
        }
    }

    tabClick = (selectedType) => {
        const self = this;
        self.sendEvent('2', `clicked on ${selectedType} tab`, 'alert', 'Alert Page');
        let { regionId, countryId, cityId, route, monthcolumns } = this.state;
        self.setState({ type: selectedType, drillDownData: [], drillDownTotalData: [], loading2: true })

        self.getDrillDownData(regionId, countryId, cityId, route, selectedType)
        this.gridApiMonth.setColumnDefs(monthcolumns);
    }

    homeHandleClick = (e) => {
        const self = this;
        this.directPOS = false
        const userDetails = JSON.parse(cookieStorage.getCookie('userDetails'));
        let access = userDetails.access;
        window.$('.nav-tabs li a[href="#Section6"]').tab('show')

        if (access === '#*') {
            self.sendEvent('2', 'clicked on Network', 'alert', 'Alert Page');

            self.setState({ loading: true, loading2: true, firstHome: false, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [], toggle: 'bc' })

            window.localStorage.setItem('RegionSelected', 'Null')
            window.localStorage.setItem('CountrySelected', 'Null')
            window.localStorage.setItem('CitySelected', 'Null')
            window.localStorage.setItem('ODSelected', 'Null')

            self.getMonthDrillDownData('*', '*', '*', '*')

            bcData = [];
            const newURL = window.location.href.split("?")[0];
            window.history.pushState('object', document.title, newURL);
            // this.props.history.push('/pos')
        }
    }

    listHandleClick = (data, title, selection) => {
        const self = this;
        self.sendEvent('2', 'clicked on Drill down list', 'alert', 'Alert Page');
        let { regionId, countryId, cityId } = this.state;
        let selectedData = data;
        if (selectedData.charAt(0) !== "'" && selectedData.charAt(selectedData.length - 1) !== "'") {
            selectedData = `'${data}'`
        }
        if ((data).includes(',')) {
            selectedData = `'${data.split(',').join("','")}'`;
        }
        self.setState({ selectedData, loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: [] })
        const getColName = decodeURIComponent(title);

        if (selection === 'List') {
            const indexEnd = bcData.findIndex(function (d) {
                return d.title === title;
            })
            const removeArrayIndex = bcData.slice(0, indexEnd + 1);
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
            window.localStorage.setItem('SelectedCabin', JSON.stringify(getCabinValue));
        })
    }

    onCabinClose() {
        const self = this;
        self.sendEvent('2', 'clicked on Cabin drop down', 'alert', 'Alert Page');
        let { cabinSelectedDropDown } = this.state;

        if (cabinSelectedDropDown.length > 0) {
            this.getDataOnCabinChange()
        } else {
            this.setState({ getCabinValue: 'Null' }, () => this.getDataOnCabinChange())
            window.localStorage.setItem('SelectedCabin', 'Null');
        }
    }

    getDataOnCabinChange() {
        const self = this;
        self.setState({
            loading: true, loading2: true, monthData: [], monthTotalData: [], drillDownData: [], drillDownTotalData: []
        })
        let { regionId, countryId, cityId, route } = this.state;
        self.getMonthDrillDownData(regionId, countryId, cityId, route)
    }

    toggle = (e) => {
        let currency = e.target.value;
        let { regionId, countryId, cityId, route } = this.state;
        this.sendEvent('2', 'clicked on Currency toggle', 'alert', 'Alert Page');
        this.setState({ toggle: currency }, () => this.getMonthDrillDownData(regionId, countryId, cityId, route))
    }

    redirection = (e) => {
        this.sendEvent('2', 'clicked on POS/Route drop down', 'alert', 'Alert Page');
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
        const element = document.createElement("span");
        let isAlert = params.data.isAlert;
        isAlert = isAlert !== undefined ? isAlert.toString() : null
        if (isAlert !== '0' && isAlert !== null) {
            // if (header !== 'Cabin' && header !== 'Agency' && header !== 'Ancillary') {
            const icon = document.createElement("i");
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

    serialize = (params) => {
        const str = [];
        for (const p in params)
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
        let { tabName } = this.state;
        console.log("Tab", tabName);

        return (
            <ul className="nav nav-tabs" role="tablist">

                <li role="presentation" onClick={() => this.tabClick('Null')} >
                    <a href="#Section1" aria-controls="home" role="tab" data-toggle="tab">
                        {tabName}
                    </a>
                </li>

                {tabName === 'Country' || tabName === "POS" || tabName === "OD" || tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('Country')}>
                    <a href="#Section3" aria-controls="messages" role="tab" data-toggle="tab">
                        Country
                    </a>
                </li>}

                {tabName === 'POS' || tabName === "OD" || tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('POS')}>
                    <a href="#Section4" aria-controls="messages" role="tab" data-toggle="tab">
                        POS
                    </a>
                </li>}
                {tabName === 'OD' || tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('OD')}>
                    <a href="#Section5" aria-controls="messages" role="tab" data-toggle="tab">
                        OD
                    </a>
                </li>}
                {tabName === 'Cabin' ? '' : <li role="presentation" onClick={() => this.tabClick('Cabin')}>
                    <a href="#Section6" aria-controls="messages" role="tab" data-toggle="tab">
                        Cabin
                    </a>
                </li>}
                <li role="presentation" className="active" onClick={() => this.tabClick('Alert')}>
                    <a href="#Section6" aria-controls="messages" role="tab" data-toggle="tab">
                        Alert
                    </a>
                </li>

            </ul>
        )
    }

    renderChart = () => {
        return (
            <ChartModelDetails
                chartVisible={this.state.chartVisible}
                displayName={this.state.chartHeader}
                gettingYear={this.state.gettingYear}
                gettingMonth={this.state.gettingMonth}
                closeChartModal={() => this.setState({ chartVisible: false })}
                alert={true}
                selectedData={this.state.selectedDataChart} 
                isDirectPOS={this.directPOS}/>
        )
    }

    gridApiMonthly = (api) => {
        this.gridApiMonth = api;
    }

    mgridApiDrillDown = (api) => {
        this.gridApiDrillDown = api
    }

    getRowStyle = (params) => {
        if (params.data.Month === "High") {
            return { background: '#ff4141' };
        } else if (params.data.Month === "Moderate") {
            return { background: '#fff500', color: "black" };
        } else if (params.data.Month === "Low") {
            return { background: '#00b03c' };
        }
    }

    render() {
        const { accessLevelDisable } = this.state;
        return (
            <div className='pos-details'>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className="row">
                    <div className="col-md-12 col-sm-12 col-xs-12 top">
                        <div className="navdesign" style={{ marginTop: '0px' }}>
                            <div className="col-md-7 col-sm-7 col-xs-7 toggle1">
                                <h3>Alert Module</h3>
                                <section>
                                    <nav>
                                        <ol className="cd-breadcrumb">
                                            <div style={{ cursor: accessLevelDisable ? 'not-allowed' : 'pointer' }}>
                                                <li className={`${accessLevelDisable ? 'breadcrumb-disable' : ''}`} onClick={() => this.homeHandleClick()}> {'Network'} </li>
                                            </div>
                                            {this.state.firstLoadList ? "" : bcData.map((item) =>
                                                <div style={{ cursor: item.disable ? 'not-allowed' : 'pointer' }}>
                                                    <li key={item.title} className={`${item.disable ? 'breadcrumb-disable' : ''}`} onClick={(e) => this.listHandleClick(e.target.id, item.title, 'List')} id={item.val} title={`${item.title} : ${item.val}`}>
                                                        {` > ${item.val}`}
                                                    </li>
                                                </div>
                                            )}
                                        </ol>
                                    </nav>
                                </section>
                            </div>

                        </div>

                    </div>
                </div>

                <div className="row alert-row">
                    <div className="col-md-12 col-sm-12 col-xs-12">
                        <div className="x_panel" style={{ marginTop: "10px", height: 'calc(100vh - 130px)' }}>
                            <div className="x_content">

                                <DataTableComponent
                                    gridApi={this.gridApiMonthly}
                                    rowData={this.state.monthData}
                                    columnDefs={this.state.monthcolumns}
                                    loading={this.state.loading}
                                    onCellClicked={(cellData) => this.monthWiseCellClick(cellData)}
                                    frameworkComponents={{ customHeaderGroupComponent: POSCustomHeaderGroup }}
                                    rowClassRules={this.state.posMonthRowClassRule}
                                    pos={true}
                                    ensureIndexVisible={this.state.ensureIndexVisible}
                                    getRowStyle={this.getRowStyle} />
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
                                        <div role="tabpanel" className="tab-pane fade in active" id="Section1">

                                            <DataTableComponent
                                                gridApi={this.mgridApiDrillDown}
                                                rowData={this.state.drillDownData}
                                                columnDefs={this.state.drillDownColumn}
                                                onCellClicked={(cellData) => this.regionCellClick(cellData)}
                                                loading={this.state.loading2}
                                                pos={true}
                                                ensureIndexVisible={this.state.ensureIndexVisibleDD}
                                                getRowStyle={this.getRowStyle} />
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
                        rowData={this.state.alertTrendData}
                        columns={this.state.alertTrendColumn}
                        interline={this.state.interline}
                        header={`${this.state.gettingMonth} ${this.state.gettingYear} ${this.state.alertTrendType ? ` - ${this.state.alertTrendType}` : ""}`}
                        loading={this.state.loading3}
                        getRowStyle={(params) => {
                            if (params.data.Alert_Type === "High") {
                                return { background: '#ff4141' };
                            } else if (params.data.Alert_Type === "Moderate") {
                                return { background: '#fff500', color: "black" };
                            } else if (params.data.Alert_Type === "Low") {
                                return { background: '#00b03c' };
                            } else if (params.data.Alert_Type === "Total") {
                                return { background: '#1784c7' };
                            }
                        }}
                        totalData={this.state.alertTrendTotalData}
                    />
                    <AlertModal
                        alertVisible={this.state.alertVisible}
                        alertData={this.state.alertData}
                        closeAlertModal={() => this.setState({ alertVisible: false })}
                    />
                </div>

                {
                    this.renderChart()
                }

            </div>

        );
    }
}

const NewComponent = BrowserToProps(AlertsNew);

export default NewComponent;