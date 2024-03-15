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
import String from '../Constants/validator';
import DatePicker from '../Component/DatePicker'
import APIServices from '../API/apiservices'
import api from '../API/api';
import './component.scss'
import TextField from '@material-ui/core/TextField';
import Autocomplete from '@material-ui/lab/Autocomplete';
import CheckBoxOutlineBlankIcon from '@material-ui/icons/CheckBoxOutlineBlank';
import CheckBoxIcon from '@material-ui/icons/CheckBox';
import { size } from 'lodash';

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

const rangeValue = {
    from: {
        year: new Date().getFullYear(),
        month: 1
    },
    to: {
        year: new Date().getFullYear(),
        month: (new Date().getMonth() + 1)
    }
}

const rangeValueNextYear = {
    from: {
        year: new Date().getFullYear() + 1,
        month: 1
    },
    to: {
        year: new Date().getFullYear() + 1,
        month: 1
    }
}
const icon = <CheckBoxOutlineBlankIcon fontSize="small" />;
const checkedIcon = <CheckBoxIcon fontSize="small" />;

class RegionsDropDown extends Component {
    constructor(props) {
        super(props);
        this.state = {
            months: [],
            gettingMonth: null,
            gettingYear: null,
            gettingMonthA: 'Null',
            gettingYearA: 'Null',
            selectedMonth: '',
            regions: [],
            regionSelected: 'Null',
            countries: [],
            countrySelected: 'Null',
            cities: [],
            citySelected: 'Null',
            ODs: [],
            ODSelected: 'Null',
            regionSelectedDropDown: [],
            countrySelectedDropDown: [],
            citySelectedDropDown: [],
            ODSelectedDropDown: [],
            rangeValuePicker: rangeValue,
            startDate: this.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1)),
            endDate: this.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0)),
            currentMonth: (new Date()).getMonth(),
            currentYear: (new Date()).getFullYear(),
            cabinOption: [],
            getCabinValue: [],
            cabinSelectedDropDown: [],
            cabinDisable: true,
            regionDisable: false,
            countryDisable: true,
            cityDisable: true,
            ODDisable: true,
            regionAccess: false,
            countryAccess: false,
            cityAccess: false,
            ODAccess: false,
            searchOD: '',
            setsearchOD: '',
        }
        this.pickRange = React.createRef()
    }


    componentWillMount() {
        var range = this.props.monthRange === 'nextYear' ? window.localStorage.getItem('rangeValueNextYear') : window.localStorage.getItem('rangeValue')
        if (range === null) {
            window.localStorage.setItem('rangeValue', JSON.stringify(this.state.rangeValuePicker))
            window.localStorage.setItem('rangeValueNextYear', JSON.stringify(rangeValueNextYear))
        } else {
            range = JSON.parse(range)
            const selectedMonth = `${window.shortMonthNumToName(range.from.month)} ${range.from.year}`
            this.setState({ selectedMonth: selectedMonth, gettingYear: range.from.year, gettingMonth: range.from.month })
        }

        apiServices.getRegions().then((regionsData) => {
            if (regionsData) {
                this.setState({ regions: regionsData })
                cookieStorage.createCookie('Regions', JSON.stringify(regionsData), 1);
            }
        })

        apiServices.getClassNameDetails().then((result) => {
            if (result && result.length > 0) {
                var classData = result[0].classDatas;
                this.setState({ cabinOption: classData, cabinDisable: false })
            }
        });

        this.setFilterValues();

    }

    setFilterValues() {
        let details = cookieStorage.getCookie('userDetails')
        details = details ? JSON.parse(details) : ''

        if (details.access !== '#*') {
            let access = details.access;
            let accessList = access.split('#');
            let RegionSelected = accessList[1]
            let CountrySelected = accessList[2] === '*' ? 'Null' : accessList[2]
            let CitySelected = accessList[2] === '*' ? 'Null' : accessList[3] === '*' ? 'Null' : accessList[3]

            let region = []
            let country = []
            let city = []
            let OD = []

            region.push(String.removeQuotes(RegionSelected))
            window.localStorage.setItem('RegionSelected', JSON.stringify(region))
            this.setState({ regionDisable: true, regionAccess: true })

            if (CountrySelected !== 'Null') {
                country.push(String.removeQuotes(CountrySelected))
                window.localStorage.setItem('CountrySelected', JSON.stringify(country))
                this.setState({ countryDisable: true, countryAccess: true })
            }

            if (CitySelected !== 'Null') {
                city.push(String.removeQuotes(CitySelected))
                window.localStorage.setItem('CitySelected', JSON.stringify(city))
                this.setState({ cityDisable: true, cityAccess: true })
            }

            this.getInitialFilterValuesFromLS()

        } else {
            this.getInitialFilterValuesFromLS()
        }
        if (Object.keys(details.route_access).length > 0) {
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
        }
    }

    getInitialFilterValuesFromLS() {
        var rangeValue = this.props.monthRange === 'nextYear' ? JSON.parse(window.localStorage.getItem('rangeValueNextYear')) : JSON.parse(window.localStorage.getItem('rangeValue'))
        let startDate = '';
        let endDate = '';
        if (rangeValue !== null) {
            startDate = this.formatDate(new Date(rangeValue.from.year, rangeValue.from.month - 1, 1));
            endDate = this.formatDate(new Date(rangeValue.to.year, rangeValue.to.month, 0));
            this.setState({ rangeValuePicker: rangeValue })
        }
        startDate = startDate === '' ? this.state.startDate : startDate
        endDate = endDate === '' ? this.state.endDate : endDate

        let regionSelected = window.localStorage.getItem('RegionSelected')
        let countrySelected = window.localStorage.getItem('CountrySelected')
        let citySelected = window.localStorage.getItem('CitySelected')
        let ODSelected = window.localStorage.getItem('ODSelected')
        let getCabinValue = window.localStorage.getItem('CabinSelected')

        let regionSelectedDropDown = regionSelected === null || regionSelected === 'Null' ? [] : JSON.parse(regionSelected);
        let countrySelectedDropDown = countrySelected === null || countrySelected === 'Null' ? [] : JSON.parse(countrySelected);
        let citySelectedDropDown = citySelected === null || citySelected === 'Null' ? [] : JSON.parse(citySelected);
        let ODSelectedDropDown = ODSelected === null || ODSelected === 'Null' ? [] : JSON.parse(ODSelected)
        let cabinSelectedDropDown = getCabinValue === null || getCabinValue === 'Null' ? [] : JSON.parse(getCabinValue);

        regionSelected = regionSelectedDropDown.length > 0 ? regionSelectedDropDown : 'Null'
        countrySelected = countrySelectedDropDown.length > 0 ? countrySelectedDropDown : 'Null'
        citySelected = citySelectedDropDown.length > 0 ? citySelectedDropDown : 'Null'
        ODSelected = ODSelectedDropDown.length > 0 ? ODSelectedDropDown : 'Null'
        getCabinValue = cabinSelectedDropDown.length > 0 ? cabinSelectedDropDown : 'Null'

        let ODtype = [];
        ODSelectedDropDown.forEach((j) => {
            ODtype.push({ 'OD': j })

        });

        this.setState({
            regionSelected: regionSelected,
            countrySelected: countrySelected,
            citySelected: citySelected,
            ODSelected: ODSelected,
            regionSelectedDropDown: regionSelectedDropDown,
            countrySelectedDropDown: countrySelectedDropDown,
            citySelectedDropDown: citySelectedDropDown,
            ODSelectedDropDown: ODtype,
            startDate: startDate,
            endDate: endDate,
            getCabinValue: getCabinValue,
            cabinSelectedDropDown: cabinSelectedDropDown
        }, () => { this.props.getFilterValues(this.state); this.getCountries() })
    }

    callRegion = (e) => {
        const selectedRegionPicker = e.target.value;

        this.setState({
            regionSelected: selectedRegionPicker,
            regionSelectedDropDown: selectedRegionPicker,
            countries: [],
            countrySelected: 'Null',
            countrySelectedDropDown: [],
            cities: [],
            citySelected: 'Null',
            citySelectedDropDown: [],
            ODs: [],
            ODSelected: 'Null',
            ODSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('RegionSelected', JSON.stringify(selectedRegionPicker));
            window.localStorage.setItem('CountrySelected', 'Null');
            window.localStorage.setItem('CitySelected', 'Null');
            window.localStorage.setItem('ODSelected', 'Null');
        })
    }

    onRegionClose() {
        this.getCountries();
        if (this.state.regions.length > 0) {
            if (this.state.regionSelectedDropDown.length === 0) {
                this.setState({ regionSelected: 'Null' })
                window.localStorage.setItem('RegionSelected', 'Null');
            }
        }
    }

    callCountry = (e) => {
        const selectedCountryPicker = e.target.value;

        this.setState({
            countrySelected: selectedCountryPicker,
            countrySelectedDropDown: selectedCountryPicker,
            cities: [],
            citySelected: 'Null',
            citySelectedDropDown: [],
            ODs: [],
            ODSelected: 'Null',
            ODSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('CountrySelected', JSON.stringify(selectedCountryPicker));
            window.localStorage.setItem('CitySelected', 'Null');
            window.localStorage.setItem('ODSelected', 'Null');
        })
    }

    onCountryClose() {
        this.getCities();
        if (this.state.countries.length > 0) {
            if (this.state.countrySelectedDropDown.length === 0) {
                this.setState({ countrySelected: 'Null' })
                window.localStorage.setItem('CountrySelected', 'Null');
            }
        }
    }

    callCity = (e) => {
        const selectedCityPicker = e.target.value;
        this.setState({
            citySelected: selectedCityPicker,
            citySelectedDropDown: selectedCityPicker,
            ODs: [],
            ODSelected: 'Null',
            ODSelectedDropDown: []
        }, () => {
            window.localStorage.setItem('CitySelected', JSON.stringify(selectedCityPicker));
            window.localStorage.setItem('ODSelected', 'Null');
        })
    }

    onCityClose() {
        this.getODs();
        if (this.state.cities.length > 0) {
            if (this.state.citySelectedDropDown.length === 0) {
                this.setState({ citySelected: 'Null' })
                window.localStorage.setItem('CitySelected', 'Null');
            }
        }
    }

    callOD = (value) => {
        const selectedODPicker = value.map((d) => d.OD);
        this.setState({
            ODSelected: selectedODPicker.length == 0 ? 'Null' : selectedODPicker,
            ODSelectedDropDown: value,
        }, () => {
            window.localStorage.setItem('ODSelected', JSON.stringify(selectedODPicker));
        })
    }

    onODClose() {
        if (this.state.ODs.length > 0) {
            if (this.state.ODSelectedDropDown.length === 0) {
                this.setState({ ODSelected: 'Null' })
                window.localStorage.setItem('ODSelected', 'Null');
            }
        }
    }


    callMonthDM = () => {
        const e = document.getElementById('monthDM')
        const gettingMonth = e.options[e.selectedIndex].text;
        const gettingYear = e.options[e.selectedIndex].value
        this.setState({
            gettingMonthA: gettingYear === 'Null' ? 'Null' : gettingMonth,
            gettingYearA: gettingYear
        })
    }

    getCountries = () => {
        this.setState({ countryDisable: true, cityDisable: true })
        apiServices.getCountries(this.state.regionSelected).then((countriesData) => {
            if (countriesData && countriesData.length > 0) {
                if (!this.state.countryAccess) {
                    this.setState({ countryDisable: false })
                }
                this.setState({ countries: countriesData }, () => this.getCities())
            }
        });
    }

    getCities = () => {
        this.setState({ cityDisable: true })
        apiServices.getCities(this.state.regionSelected, this.state.countrySelected).then((citiesData) => {
            if (citiesData && citiesData.length > 0) {
                if (!this.state.cityAccess) {
                    this.setState({ cityDisable: false })
                }
                this.setState({ cities: citiesData }, () => this.getODs())
            }
        })
    }

    debounce = (func, timeout = 300) => {
        let timer;
        return (...args) => {
            clearTimeout(timer);
            timer = setTimeout(() => { func.apply(this, args); }, timeout);
        };
    }

    // processChange = this.debounce((e) => this.handleChange(e));

    handleChange = (e) => {
        let searchOD = e.target.value;
        api.get(`getCommonODByCity?regionId=${String.addQuotesforMultiSelect(this.state.regionSelected)}&countryCode=${String.addQuotesforMultiSelect(this.state.countrySelected)}&cityCode=${String.addQuotesforMultiSelect(this.state.citySelected)}&searchod=${(searchOD)}`)
            .then((response) => {
                this.setState({ ODs: response.data })
            })
    }

    getODs = () => {
        this.setState({ ODDisable: true })
        apiServices.getODs(this.state.regionSelected, this.state.countrySelected, this.state.citySelected).then((ODsData) => {
            if (ODsData && ODsData.length > 0) {
                if (!this.state.ODAccess) {
                    this.setState({ ODDisable: false })
                }
                this.setState({ ODs: ODsData })
            }
        })
    }

    handleRangeDissmis = (value) => {
        let startDate = this.formatDate(new Date(value.from.year, value.from.month - 1, 1));
        let endDate = this.formatDate(new Date(value.to.year, value.to.month, 0));
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

    formatDate = (d) => {
        let month = '' + (d.getMonth() + 1);
        let day = '' + d.getDate();
        let year = d.getFullYear();

        if (month.length < 2)
            month = '0' + month;
        if (day.length < 2)
            day = '0' + day;

        return [year, month, day].join('-');
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
        let { cabinSelectedDropDown } = this.state;

        if (cabinSelectedDropDown.length > 0) {
            let _ = this.props.dashboard ? this.props.getFilterValues(this.state) : null
        } else {
            this.setState({ getCabinValue: 'Null' }, () => {
                let _ = this.props.dashboard ? this.props.getFilterValues(this.state) : null
            })
            window.localStorage.setItem('CabinSelected', 'Null');
        }
    }

    search() {
        this.props.getFilterValues(this.state)
    }

    getNext3Months() {
        const { currentMonth, currentYear } = this.state;
        let year = currentYear;
        let monthsArray = [1, 2, 3, 4].map(n => (currentMonth + n) % 12)// %12 caters for end of year wrap-around.
        const next3Months = monthsArray.map((d, i) => {
            if (d === 0) {
                d = 12
            }
            if (monthsArray[i - 1] === 0 || monthsArray[i - 2] === 0 || monthsArray[i - 3] === 0) {
                year = currentYear + 1
            }
            return { 'Month': window.monthNumToName(d), 'Year': year };
        })
        return next3Months;
    }

    render() {
        const { agentDashboard, dashboard, hideCabin, pageName, BoughtDropout, DemoDomInt, Demography } = this.props;

        const { regions, regionSelectedDropDown, countrySelectedDropDown, citySelectedDropDown,
            countries, cities, months, selectedMonth, cabinOption, cabinSelectedDropDown,
            cabinDisable, countryDisable, cityDisable, ODs, ODDisable, ODSelectedDropDown, rangeValuePicker, pickerLang, regionDisable, currentYear, searchod, r } = this.state;

        const monthOptionItems = months.map((month) =>
            <option selected={selectedMonth === month.Month} id={month.Year} value={month.MonthName}>{month.Month}</option>
        );

        return (
            <div className={`regions-dropdown`}>
                {!agentDashboard ?
                    <DatePicker
                        rangeValue={rangeValuePicker}
                        handleRangeDissmis={this.handleRangeDissmis}
                        {...this.props}
                    />
                    : ''}
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
                                <MenuItem key={r.RegionID} value={r.Region}>
                                    <Checkbox checked={regionSelectedDropDown.indexOf(r.Region) > -1} />
                                    <ListItemText primary={r.Region} />
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
                                <MenuItem key={r.CountryCode} value={r.CountryCode}>
                                    <Checkbox checked={countrySelectedDropDown.indexOf(r.CountryCode) > -1} />
                                    <ListItemText primary={r.CountryCode} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>

                {BoughtDropout || DemoDomInt || Demography ? '' : <div className={`${citySelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                    <FormControl className={`select-group ${cityDisable ? 'disable-main' : ''}`}>
                        <InputLabel id="demo-mutiple-checkbox-label">City</InputLabel>
                        <Select
                            labelId="demo-mutiple-checkbox-label"
                            id="demo-mutiple-checkbox"
                            className={`${cityDisable ? 'disable' : ''}`}
                            multiple
                            value={citySelectedDropDown}
                            onChange={(e) => this.callCity(e)}
                            input={<Input />}
                            renderValue={selected => {
                                return selected.join(',')
                            }}
                            onClose={() => this.onCityClose()}
                        >
                            {cities.map(r => (
                                <MenuItem key={r.CityCode} value={r.CityCode}>
                                    <Checkbox checked={citySelectedDropDown.indexOf(r.CityCode) > -1} />
                                    <ListItemText primary={r.CityCode} />
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                </div>}

                {pageName == 'top_markets' ?
                    <div className={`${ODSelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                        <FormControl className={`select-group ${ODDisable ? 'disable-main' : ''}`}>
                            <Autocomplete
                                multiple
                                id="checkboxes-tags-demo"
                                className={`${ODDisable ? 'disable' : ''}`}
                                options={ODs}
                                onChange={(event, value) => this.callOD(value)} // prints the selected value
                                disableCloseOnSelect
                                value={ODSelectedDropDown}
                                getOptionLabel={(option) => option.OD}
                                renderOption={(option, { selected }) => (
                                    <React.Fragment>
                                        <Checkbox
                                            icon={icon}
                                            checkedIcon={checkedIcon}
                                            style={{ marginRight: 8 }}
                                            checked={selected}
                                        />
                                        {option.OD}
                                    </React.Fragment>
                                )}
                                style={{ maxWidth: 300, maxHeight: 100, hmarginRight: 14, }}
                                renderInput={(params) => (
                                    <TextField
                                        {...params}
                                        variant="outlined"
                                        placeholder="OD"
                                        onChange={(e) => this.handleChange(e)}
                                    />
                                )}
                            />
                        </FormControl>
                    </div> : null}
                {/* {
                    !dashboard ? <div className="" >
                        <div className="select-group" style={{ width: 'auto' }}>
                            <h4>Month:</h4>
                            <select className="form-control cabinselect" id='months'
                                onChange={() => this.callMonth()}>
                                {monthOptionItems}
                            </select>
                        </div>
                    </div> : ''
                } */}

                {!dashboard && !hideCabin ? <div className={`${cabinSelectedDropDown.length > 0 ? 'hide-regions-dropdown-label' : ''}`} >
                    <div className="select-group" style={{ width: 'auto' }}>
                        <h4>Cabin :</h4>
                        <FormControl className={`select-group ${cabinDisable ? 'disable-main' : ''}`}>
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
                </div> : ''}

                {agentDashboard ?
                    <div className="" >
                        <div className="form-group" style={{ display: 'flex', marginBottom: '0px' }}>
                            <select className="form-control cabinselect dashboard-dropdown" style={{ marginRight: '10px', width: 'auto' }}
                                onChange={() => this.callMonthDM()} id="monthDM">
                                <option value='Null'>Select Month</option>
                                {this.getNext3Months().map((d, i) => <option value={d.Year}>{d.Month}</option>)}
                            </select>
                        </div>
                    </div>
                    : ''}

                {!agentDashboard && !dashboard ? <div className="">
                    <button type="button" className="btn search" onClick={() => this.search()}>Search</button>
                </div> : <i className='fa fa-search' onClick={() => this.search()}></i>}

            </div >
        );
    }
}

export default RegionsDropDown;