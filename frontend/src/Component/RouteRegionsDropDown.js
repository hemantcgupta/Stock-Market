import React, { Component } from 'react';
import { makeStyles } from "@material-ui/core/styles";
import Input from "@material-ui/core/Input";
import InputLabel from "@material-ui/core/InputLabel";
import MenuItem from "@material-ui/core/MenuItem";
import FormControl from "@material-ui/core/FormControl";
import ListItemText from "@material-ui/core/ListItemText";
import Select from "@material-ui/core/Select";
import Checkbox from "@material-ui/core/Checkbox";
import cookieStorage from '../Constants/cookie-storage'
import Constant from '../Constants/validator'
import DatePicker from '../Component/DatePicker'
import DateRangePicker from '../Component/DateRangePicker'
import { subDays } from 'date-fns';
import APIServices from '../API/apiservices'
import './component.scss'

const apiServices = new APIServices();

const ITEM_HEIGHT = 48;
const ITEM_PADDING_TOP = 8;
const MenuProps = {
    PaperProps: {
        style: {
            maxHeight: ITEM_HEIGHT * 4.5 + ITEM_PADDING_TOP,
            width: 250
        }
    }
};

const currentYear = new Date().getFullYear()

const rangeValue = {
    from: {
        year: currentYear,
        month: 1
    },
    to: {
        year: currentYear,
        month: (new Date().getMonth() + 1)
    }
}

const rangeValueNextYear = {
    from: {
        year: currentYear + 1,
        month: 1
    },
    to: {
        year: currentYear + 1,
        month: 1
    }
}

const currentDate = Constant.getDateFormat(new Date())
const first5DaysofYear = [`Jan 1, ${currentYear}`, `Jan 2, ${currentYear}`, `Jan 3, ${currentYear}`, `Jan 4, ${currentYear}`, `Jan 5, ${currentYear}`]
let ytdYearRR = currentYear
first5DaysofYear.forEach((d) => d === currentDate ? ytdYearRR = ytdYearRR - 1 : null)

class RouteRegionsDropDown extends Component {
    constructor(props) {
        super(props);
        this.state = {
            gettingMonth: null,
            gettingYear: null,
            getDay: null,
            routeGroup: 'Network',
            regions: [],
            regionSelected: 'Null',
            countries: [],
            countrySelected: 'Null',
            Routes: [],
            routeSelected: 'Null',
            regionSelectedDropDown: [],
            countrySelectedDropDown: [],
            routeSelectedDropDown: [],
            rangeValuePicker: rangeValue,
            startDate: Constant.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1)),
            endDate: Constant.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0)),
            typeofCost: '',
            regionDisable: true,
            regionAccess: false,
            countryDisable: true,
            countryAccess: false,
            RouteDisable: true,
            routesAcess: false,
            disableRouteGroup: false,
            showDatePicker: false,
            date: '',
            datePickerValue: [{
                startDate: new Date(ytdYearRR, 0, 1),
                endDate: subDays(new Date(), 5),
                key: 'selection'
            }],
            costTypes: ['TC', 'DOC', 'VC', 'Surplus/Deficit', 'Total Revenue']
        }
        this.pickRange = React.createRef()
    }

    componentWillMount() {
        const { dateRange, dashboardName } = this.props;
        const { rangeValuePicker, datePickerValue } = this.state;
        var range = dashboardName === 'Route Profitability' ? window.localStorage.getItem('RRDateRangeValue') : window.localStorage.getItem('rangeValue')
        if (range === null) {
            let RRDateRange = [{
                startDate: Constant.formatDate(datePickerValue[0].startDate),
                endDate: Constant.formatDate(datePickerValue[0].endDate),
                key: 'selection'
            }]
            window.localStorage.setItem('rangeValue', JSON.stringify(rangeValuePicker))
            window.localStorage.setItem('RRDateRangeValue', JSON.stringify(RRDateRange))
        } else {
            range = JSON.parse(range)
            if (dashboardName === 'Route Profitability') {
                const endDate = (range[0].endDate).split('-')
                this.setState({ gettingYear: parseInt(endDate[0]), gettingMonth: parseInt(endDate[1]), getday: parseInt(endDate[2]) })
            } else {
                const selectedMonth = `${window.shortMonthNumToName(range.from.month)} ${range.from.year}`
                this.setState({ selectedMonth: selectedMonth, gettingYear: range.from.year, gettingMonth: range.from.month })
            }
        }
        this.setFilterValues();
    }

    setFilterValues() {
        let details = JSON.parse(cookieStorage.getCookie('userDetails'))

        if (details.route_access === {}) {
            let routeAccess = {};
            this.getInitialFilterValuesFromLS()

        } else {
            let routeAccess = details.route_access;

            if ((routeAccess).hasOwnProperty('selectedRouteGroup')) {
                window.localStorage.setItem('RouteGroupSelected', JSON.stringify(routeAccess['selectedRouteGroup']))
                this.setState({ disableRouteGroup: true })
            }
            if ((routeAccess).hasOwnProperty('selectedRouteRegion')) {
                window.localStorage.setItem('RouteRegionSelected', JSON.stringify(routeAccess['selectedRouteRegion']))
                this.setState({ regionDisable: true, regionAccess: true })
            }
            if ((routeAccess).hasOwnProperty('selectedRouteCountry')) {
                window.localStorage.setItem('RouteCountrySelected', JSON.stringify(routeAccess['selectedRouteCountry']))
                this.setState({ countryDisable: true, countryAccess: true })
            }
            if ((routeAccess).hasOwnProperty('selectedRoute')) {
                window.localStorage.setItem('RouteSelected', JSON.stringify(routeAccess['selectedRoute']))
                this.setState({ RouteDisable: true, routesAcess: true })
            }

            this.getInitialFilterValuesFromLS()
        }
        if (details.access !== '#*') {
            let access = details.access;
            let accessList = access.split('#');
            let RegionSelected = accessList[1]
            let CountrySelected = accessList[2] === '*' ? 'Null' : accessList[2]
            let CitySelected = accessList[2] === '*' ? 'Null' : accessList[3] === '*' ? 'Null' : accessList[3]

            let region = []
            let country = []
            let city = []

            region.push(RegionSelected)
            window.localStorage.setItem('RegionSelected', JSON.stringify(region))

            if (CountrySelected !== 'Null') {
                country.push(CountrySelected)
                window.localStorage.setItem('CountrySelected', JSON.stringify(country))
            }

            if (CitySelected !== 'Null') {
                city.push(CitySelected)
                window.localStorage.setItem('CitySelected', JSON.stringify(city))
            }
        }
    }

    getInitialFilterValuesFromLS() {
        this.getDateRanges();
        let routeGroup = window.localStorage.getItem('RouteGroupSelected')
        let regionSelected = window.localStorage.getItem('RouteRegionSelected')
        let countrySelected = window.localStorage.getItem('RouteCountrySelected')
        let routeSelected = window.localStorage.getItem('RouteSelected')
        let costTypeSelected = window.localStorage.getItem('CostTypeSelected')

        routeGroup = routeGroup === null || routeGroup === 'Null' ? ['Network'] : JSON.parse(routeGroup);
        let regionSelectedDropDown = regionSelected === null || regionSelected === 'Null' ? [] : JSON.parse(regionSelected);
        let countrySelectedDropDown = countrySelected === null || countrySelected === 'Null' ? [] : JSON.parse(countrySelected);
        let routeSelectedDropDown = routeSelected === null || routeSelected === 'Null' ? [] : JSON.parse(routeSelected);

        regionSelected = regionSelectedDropDown.length > 0 ? regionSelectedDropDown : 'Null'
        countrySelected = countrySelectedDropDown.length > 0 ? countrySelectedDropDown : 'Null'
        routeSelected = routeSelectedDropDown.length > 0 ? routeSelectedDropDown : 'Null'

        this.setState({
            routeGroup: `'${routeGroup.join("','")}'`,
            regionSelected: regionSelected,
            countrySelected: countrySelected,
            routeSelected: routeSelected,
            regionSelectedDropDown: regionSelectedDropDown,
            countrySelectedDropDown: countrySelectedDropDown,
            routeSelectedDropDown: routeSelectedDropDown,
            typeofCost: costTypeSelected === null ? 'TC' : costTypeSelected
        }, () => { this.props.getRouteFilterValues(this.state); this.getRouteRegions() })
    }

    getDateRanges() {
        let rangeValue = null
        let startDate = '';
        let endDate = '';
        if (this.props.dashboardName === 'Route Profitability') {
            rangeValue = JSON.parse(window.localStorage.getItem('RRDateRangeValue'));
            if (rangeValue !== null) {
                startDate = rangeValue[0].startDate;
                endDate = rangeValue[0].endDate;
                let RRDateRange = [{
                    startDate: new Date(startDate.split('-')[0], startDate.split('-')[1] - 1, startDate.split('-')[2]),
                    endDate: new Date(endDate.split('-')[0], endDate.split('-')[1] - 1, endDate.split('-')[2]),
                    key: 'selection'
                }]
                this.setState({
                    datePickerValue: RRDateRange,
                    date: `${Constant.getDateFormat(RRDateRange[0].startDate)} - ${Constant.getDateFormat(RRDateRange[0].endDate)}`
                })
            }
            startDate = startDate === '' ? Constant.formatDate(this.state.datePickerValue[0].startDate) : startDate
            endDate = endDate === '' ? Constant.formatDate(this.state.datePickerValue[0].endDate) : endDate
        } else {
            rangeValue = JSON.parse(window.localStorage.getItem('rangeValue'));
            if (rangeValue !== null) {
                startDate = Constant.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1));
                endDate = Constant.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0));
                this.setState({ rangeValuePicker: rangeValue })
            }
            startDate = startDate === '' ? this.state.startDate : startDate
            endDate = endDate === '' ? this.state.endDate : endDate
        }

        this.setState({ startDate: startDate, endDate: endDate })
    }

    callRouteGroup(e) {
        let routeGroup = []
        routeGroup.push(e.target.value)
        window.localStorage.setItem('RouteGroupSelected', JSON.stringify(routeGroup))
        this.setState({
            routeGroup: `'${e.target.value}'`,
            regions: [],
            regionSelected: 'Null',
            regionSelectedDropDown: [],
            countries: [],
            countrySelected: 'Null',
            countrySelectedDropDown: [],
            Routes: [],
            routeSelected: 'Null',
            routeSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('RouteRegionSelected', 'Null');
            window.localStorage.setItem('RouteCountrySelected', 'Null');
            window.localStorage.setItem('RouteSelected', 'Null');
            window.localStorage.setItem('LegSelected', 'Null');
            window.localStorage.setItem('FlightSelected', 'Null');
            window.localStorage.setItem('RPFlightSelected', 'Null');
            this.getRouteRegions()
        })
    }

    callRegion = (e) => {
        const selectedRegionPicker = e.target.value;

        this.setState({
            regionSelected: selectedRegionPicker,
            regionSelectedDropDown: selectedRegionPicker,
            countries: [],
            countrySelected: 'Null',
            countrySelectedDropDown: [],
            Routes: [],
            routeSelected: 'Null',
            routeSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('RouteRegionSelected', JSON.stringify(selectedRegionPicker));
            window.localStorage.setItem('RouteCountrySelected', 'Null');
            window.localStorage.setItem('RouteSelected', 'Null');
            window.localStorage.setItem('LegSelected', 'Null');
            window.localStorage.setItem('FlightSelected', 'Null');
            window.localStorage.setItem('RPFlightSelected', 'Null');
        })
    }

    onRegionClose() {
        this.getRouteCountries();
        if (this.state.regions.length > 0) {
            if (this.state.regionSelectedDropDown.length === 0) {
                this.setState({ regionSelected: 'Null' })
                window.localStorage.setItem('RouteRegionSelected', 'Null');
            }
        }
    }

    callCountry = (e) => {
        const selectedCountryPicker = e.target.value;

        this.setState({
            countrySelected: selectedCountryPicker,
            countrySelectedDropDown: selectedCountryPicker,
            Routes: [],
            routeSelected: 'Null',
            routeSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('RouteCountrySelected', JSON.stringify(selectedCountryPicker));
            window.localStorage.setItem('RouteSelected', 'Null');
            window.localStorage.setItem('LegSelected', 'Null');
            window.localStorage.setItem('FlightSelected', 'Null');
            window.localStorage.setItem('RPFlightSelected', 'Null');
        })
    }

    onCountryClose() {
        this.getRoutes();
        if (this.state.countries.length > 0) {
            if (this.state.countrySelectedDropDown.length === 0) {
                this.setState({ countrySelected: 'Null' })
                window.localStorage.setItem('RouteCountrySelected', 'Null');
            }
        }
    }

    callRoute = (e) => {
        const selectedRoutePicker = e.target.value;

        this.setState({
            routeSelected: selectedRoutePicker,
            routeSelectedDropDown: selectedRoutePicker,
        }, () => {
            window.localStorage.setItem('RouteSelected', JSON.stringify(selectedRoutePicker));
            window.localStorage.setItem('LegSelected', 'Null');
            window.localStorage.setItem('FlightSelected', 'Null');
            window.localStorage.setItem('RPFlightSelected', 'Null');
        })
    }

    onRouteClose() {
        if (this.state.Routes.length > 0) {
            if (this.state.routeSelectedDropDown.length === 0) {
                this.setState({ routeSelected: 'Null' })
                window.localStorage.setItem('RouteSelected', 'Null');
            }
        }
    }

    getRouteRegions = () => {
        this.setState({ regionDisable: true, countryDisable: true, RouteDisable: true })
        apiServices.getRouteRegions(this.state.routeGroup).then((regionsData) => {
            if (regionsData) {
                if (!this.state.regionAccess) {
                    this.setState({ regionDisable: false })
                }
                this.setState({ regions: regionsData }, () => this.getRouteCountries())
                cookieStorage.createCookie('Regions', JSON.stringify(regionsData), 1);
            }
        })
    }

    getRouteCountries = () => {
        this.setState({ countryDisable: true, RouteDisable: true })
        apiServices.getRouteCountries(this.state.routeGroup, this.state.regionSelected).then((countriesData) => {
            if (countriesData && countriesData.length > 0) {
                if (!this.state.countryAccess) {
                    this.setState({ countryDisable: false })
                }
                this.setState({ countries: countriesData }, () => this.getRoutes())
            }
        });
    }

    getRoutes = () => {
        this.setState({ RouteDisable: true })
        apiServices.getRoutes(this.state.routeGroup, this.state.regionSelected, this.state.countrySelected).then((RoutesData) => {
            if (RoutesData && RoutesData.length > 0) {
                if (!this.state.routesAcess) {
                    this.setState({ RouteDisable: false })
                }
                this.setState({ Routes: RoutesData })
            }
        })
    }

    handleRangeDissmis = (value) => {
        const { dateRange } = this.props;
        let startDate = Constant.formatDate(new Date(value.from.year, value.from.month - 1, 1));
        let endDate = Constant.formatDate(new Date(value.to.year, value.to.month, 0));
        this.setState({
            rangeValuePicker: value,
            startDate: startDate,
            endDate: endDate
        })
        if (this.props.monthRange === 'nextYear') {
            window.localStorage.setItem('rangeValueNextYear', JSON.stringify(value))
        } else {
            window.localStorage.setItem('rangeValue', JSON.stringify(value))
        }
    }

    callCostTypes = (e) => {
        let typeofCost = e.target.value;
        window.localStorage.setItem('CostTypeSelected', typeofCost)
        this.setState({ typeofCost })
    }

    search() {
        this.props.getRouteFilterValues(this.state)
    }

    handleDatePicker = (item) => {
        const startDate = Constant.getDateFormat([item.selection][0].startDate);
        const endDate = Constant.getDateFormat([item.selection][0].endDate);
        this.setState({ datePickerValue: [item.selection], date: `${startDate} - ${endDate}`, })
    }

    DatePickerClose() {
        let datePickerValue = [{ startDate: new Date(ytdYearRR, 0, 1), endDate: subDays(new Date(), 5), key: 'selection' }]
        console.log(datePickerValue[0].startDate, datePickerValue[0].endDate, Constant.getDateFormat(datePickerValue[0].startDate), Constant.formatDate(datePickerValue[0].startDate), 'startDate')
        this.setState({
            datePickerValue: datePickerValue,
            date: Constant.getDateFormat(datePickerValue[0].startDate) + '-' + Constant.getDateFormat(datePickerValue[0].endDate)
        }, () => { this.CloseDatePicker() })
    }

    CloseDatePicker() {
        this.setState({ showDatePicker: false })
    }

    dateSelected() {
        const datePickerValue = this.state.datePickerValue;
        const startDate = Constant.formatDate(datePickerValue[0].startDate)
        const endDate = Constant.formatDate(datePickerValue[0].endDate)
        this.setState({
            showDatePicker: false,
            startDate: startDate,
            endDate: endDate
        })

        const year = parseInt(endDate.split('-')[0])
        const month = parseInt(endDate.split('-')[1])
        const currentMonth = new Date(subDays(new Date(), 5)).getMonth() + 1
        const currentYear = new Date(subDays(new Date(), 5)).getFullYear()
        let RRDateRange = [{
            startDate: startDate,
            endDate: endDate,
            key: 'selection',
        }]
        if (year === currentYear) {
            if (month > currentMonth) {
                RRDateRange[0].withoutDropDown = true
            }
        } else if (year > currentYear) {
            if (month < currentMonth) {
                RRDateRange[0].withoutDropDown = true
            }
        }
        window.localStorage.setItem('RRDateRangeValue', JSON.stringify(RRDateRange))
    }


    render() {
        const { agentDashboard, dashboard, dashboardName } = this.props;
        const { regions, regionSelectedDropDown, countrySelectedDropDown, routeSelectedDropDown,
            countries, Routes, regionDisable, countryDisable, RouteDisable, rangeValuePicker, routeGroup, costTypes, typeofCost } = this.state;

        return (
            <div className='regions-dropdown'>

                {dashboardName === 'Route Profitability' ?
                    <div className="" >
                        <div className="form-group" style={{ display: 'flex', marginBottom: '0px', flexDirection: 'column' }}>
                            <button className="form-control cabinselect dashboard-dropdown date-picker-btn" onClick={() => this.setState({ showDatePicker: !this.state.showDatePicker })}>{this.state.date}</button>
                            {this.state.showDatePicker ? <div className='triangle-component'><div className="triangle-up"></div></div> : <div />}
                            <DateRangePicker
                                showDatePicker={this.state.showDatePicker}
                                dateSelected={() => this.dateSelected()}
                                onClose={() => this.DatePickerClose()}
                                handleDatePicker={item => this.handleDatePicker(item)}
                                datePickerValue={this.state.datePickerValue}
                            />
                        </div>
                    </div>
                    :
                    <DatePicker
                        rangeValue={rangeValuePicker}
                        handleRangeDissmis={this.handleRangeDissmis}
                        dateRange={this.props.dateRange}
                        {...this.props}
                    />}

                <div className="select-group" style={{ width: 'auto', marginRight: '5px' }}>
                    <select className="form-control cabinselect" id='routegroup'
                        onChange={(e) => this.callRouteGroup(e)} disabled={this.state.disableRouteGroup}>
                        <option value='Network'>Network</option>
                        <option value='Domestic' selected={routeGroup === `'Domestic'`}>Domestic</option>
                        <option value='International' selected={routeGroup === `'International'`}>International</option>
                    </select>
                </div>

                <div className={`${regionSelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                    <FormControl className={`select-group ${regionDisable ? 'disable-main' : ''}`}>
                        <InputLabel id="demo-mutiple-checkbox-label">Region</InputLabel>
                        <Select
                            labelId="demo-mutiple-checkbox-label"
                            id="demo-mutiple-checkbox"
                            className={`${regionDisable ? 'disable' : ''}`}
                            multiple
                            value={regionSelectedDropDown}
                            onChange={(e) => this.callRegion(e)}
                            input={<Input />}
                            renderValue={selected => {
                                return selected.join(',')
                            }}
                            onClose={() => this.onRegionClose()}
                        >
                            {regions.map(r => (
                                <MenuItem key={r.Route_Region} value={r.Route_Region}>
                                    <Checkbox checked={regionSelectedDropDown.indexOf(r.Route_Region) > -1} />
                                    <ListItemText primary={r.Route_Region} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>

                <div className={`${countrySelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                    <FormControl className={`select-group ${countryDisable ? 'disable-main' : ''}`}>
                        <InputLabel id="demo-mutiple-checkbox-label">Country</InputLabel>
                        <Select
                            labelId="demo-mutiple-checkbox-label"
                            id="demo-mutiple-checkbox"
                            className={`${countryDisable ? 'disable' : ''}`}
                            multiple
                            value={countrySelectedDropDown}
                            onChange={(e) => this.callCountry(e)}
                            input={<Input />}
                            renderValue={selected => {
                                return selected.join(',')
                            }}
                            onClose={() => this.onCountryClose()}
                        >
                            {countries.map(r => (
                                <MenuItem key={r.Route_Country} value={r.Route_Country}>
                                    <Checkbox checked={countrySelectedDropDown.indexOf(r.Route_Country) > -1} />
                                    <ListItemText primary={r.Route_Country} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>

                <div className={`${routeSelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                    <FormControl className={`select-group ${RouteDisable ? 'disable-main' : ''}`}>
                        <InputLabel id="demo-mutiple-checkbox-label">Route</InputLabel>
                        <Select
                            labelId="demo-mutiple-checkbox-label"
                            id="demo-mutiple-checkbox"
                            className={`${RouteDisable ? 'disable' : ''}`}
                            multiple
                            value={routeSelectedDropDown}
                            onChange={(e) => this.callRoute(e)}
                            input={<Input />}
                            renderValue={selected => {
                                return selected.join(',')
                            }}
                            onClose={() => this.onRouteClose()}
                        >
                            {Routes.map(r => (
                                <MenuItem key={r.Route} value={r.Route}>
                                    <Checkbox checked={routeSelectedDropDown.indexOf(r.Route) > -1} />
                                    <ListItemText primary={r.Route} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>

                {dashboardName === 'Route Profitability' ?
                    <div className="" >
                        <div className="select-group" style={{ width: 'auto', marginRight: '5px' }}>
                            <select className="form-control cabinselect"
                                onChange={(e) => this.callCostTypes(e)}>
                                {costTypes.map((costType) => <option value={costType} selected={typeofCost === costType}>{costType}</option>)}
                            </select>
                        </div>
                    </div> : ''}

                <i className='fa fa-search' onClick={() => this.search()}></i>
            </div >
        );
    }
}

export default RouteRegionsDropDown;