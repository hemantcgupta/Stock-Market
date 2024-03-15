import eventApi from '../API/eventApi';
import cookieStorage from '../Constants/cookie-storage';
import { subDays } from 'date-fns';

const sendEvent = (id, description, path, page_name) => {
    var eventData = {
        event_id: `${id}`,
        description: `User ${description}`,
        where_path: `/${path}`,
        page_name: `${page_name}`
    }
    eventApi.sendEvent(eventData)
}

const validateUsername = (username) => {
    let usernameRegex = /^[a-zA-Z0-9]+([._]?[a-zA-Z0-9]+)*$/;
    return usernameRegex.test(username);
}

const validateEmail = (email) => {
    let emailRegex = /^(([^<>()\[\]\\.,;:\s@"]+(\.[^<>()\[\]\\.,;:\s@"]+)*)|(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
    return emailRegex.test(email);
}

const validatePassword = (password) => {
    let passwordRegex = /^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d@$!%*#?&]{8,}$/;
    return passwordRegex.test(password);
}

const validatePhone = (phone) => {
    let phoneRegex = /(([+][(]?[0-9]{1,3}[)]?)|([(]?[0-9]{4}[)]?))\s*[)]?[-\s\.]?[(]?[0-9]{1,3}[)]?([-\s\.]?[0-9]{3})([-\s\.]?[0-9]{3,4})/;
    return phoneRegex.test(phone);
}

const validateNum = (num) => {
    let numRegex = /(\d+(?:\.\d+)?)/;
    return numRegex.test(num);
    console.log("::::REGEX::::", numRegex.test(num))
}

const validateCharNum = (char) => {
    let charRegex = /^[A-Za-z0-9]*$/;
    return charRegex.test(char);
}

const validateCity = (city) => {
    let cityRegex = /^[a-zA-Z]+(?:[\s-][a-zA-Z]+)*$/;
    return cityRegex.test(city);
}

const loggedinUser = (userDetails) => {
    const rpsConfig = userDetails.rps_config;
    return {
        userDetails: userDetails,
        accessLevelPOS: parseInt(rpsConfig.rpslevel),
        isPOSNetworkAdmin: rpsConfig.rpslevel === '4' ? true : false,
        isPOSRegionAdmin: rpsConfig.rpslevel === '3' ? true : false,
        isPOSCountryAdmin: rpsConfig.rpslevel === '2' ? true : false,
        isPOSAdmin: rpsConfig.rpslevel === '1' ? true : false,
        canEditRPS: rpsConfig.canEdit === 'TRUE' ? true : false,
        canApproveRPS: rpsConfig.canApprove === 'TRUE' ? true : false,
        canRejectRPS: rpsConfig.canReject === 'TRUE' ? true : false
    }
}

const removeQuotes = (string) => {
    let s = string !== null && string !== undefined ? typeof string === 'string' ? string : string.join(',') : ''
    if (s.charAt(0) === "'" && s.charAt(s.length - 1) === "'") {
        s = s.substring(1, s.length - 1)
    } else {
        s = s;
    }
    return s;
}

const addQuotesforMultiSelect = (string) => {
    let s = string !== null && string !== undefined ? typeof string === 'string' ? encodeURIComponent(string) : `'${encodeURIComponent(string.join("','"))}'` : ''
    return s;
}

const getParameterByName = (name, url) => {
    if (!url) url = window.location.href;
    name = name.replace(/[\[\]]/g, '\\$&');
    let regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)'),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, ' '));
}

const normalizeLineChartData = (d, delta_type) => {
    var data = d.map((el) => {
        var o = Object.assign({}, el);
        o.DataType = 'Actuals';
        return o;
    })
    let CY = data.filter((d) => d.Category === 'CY')
    let LY = data.filter((d) => d.Category === 'LY')

    CY = CY.filter((data, index) => {
        let delta = '+0'
        if (index) {
            if (delta_type === 'abs') {
                delta = CY[index].value - CY[index - 1].value
                delta = delta > 0 ? window.positiveDeltaFormat(delta) : window.negativeDeltaFormat(delta)
                delta = !delta.toString().includes('-') ? `+${delta.toString()}` : delta.toString()
            } else {
                if (CY[index].value !== 0 && CY[index - 1].value !== 0) {
                    let deltaVariance = CY[index].value - CY[index - 1].value
                    delta = window.numberFormat((deltaVariance / CY[index].value) * 100)
                    delta = !delta.toString().includes('-') ? `+${delta.toString()}%` : `${delta.toString()}%`
                }
            }
            data['delta'] = delta
        } else {
            data['delta'] = delta
        }
        return data
    })

    LY = LY.filter((data, index) => {
        let deltaLY = '+0'
        if (index) {
            if (delta_type === 'abs') {
                deltaLY = LY[index].value - LY[index - 1].value
                deltaLY = deltaLY > 0 ? window.positiveDeltaFormat(deltaLY) : window.negativeDeltaFormat(deltaLY)
                deltaLY = !deltaLY.toString().includes('-') ? `+${deltaLY.toString()}` : deltaLY.toString()
            } else {
                if (LY[index].value !== 0 && LY[index - 1].value !== 0) {
                    let deltaLYVariance = LY[index].value - LY[index - 1].value
                    deltaLY = window.numberFormat((deltaLYVariance / LY[index].value) * 100)
                    deltaLY = !deltaLY.toString().includes('-') ? `+${deltaLY.toString()}%` : `${deltaLY.toString()}%`
                }
            }
            data['delta'] = deltaLY
        } else {
            data['delta'] = deltaLY
        }
        return data
    })

    return [...CY, ...LY]
}

const rectifiedDeltaData = (d, selectedType) => {
    let data = d.map((el) => {
        var o = Object.assign({}, el);
        o.DataType = 'Actuals';
        return o;
    })

    data = data.filter((data, index) => {
        let delta_abs = null
        let delta_percent = null
        if (selectedType === 'actual') {
            delta_abs = data.delta_absolute
            delta_abs = delta_abs > 0 ? window.positiveDeltaFormat(delta_abs) : window.negativeDeltaFormat(delta_abs)
            delta_abs = !delta_abs.toString().includes('-') ? `+${delta_abs.toString()}` : delta_abs.toString()
            data['delta_absolute'] = delta_abs
            delta_percent = data.delta_percent
            delta_percent = delta_percent > 0 ? window.positiveDeltaFormat(delta_percent) : window.negativeDeltaFormat(delta_percent)
            delta_percent = !delta_percent.toString().includes('-') ? `+${delta_percent.toString()}%` : `${delta_percent.toString()}%`
            data['delta_percent'] = delta_percent
        }
        return data
    })
    return data;
}

const normalizeData = (d) => {
    var data = d.map((el) => {
        var o = Object.assign({}, el);
        o.DataType = 'Actuals';
        return o;
    })
    return data;
}

const cardsArrowIndicator = (variance) => {
    let VLY = variance.toString();
    if (VLY === '0') {
        return ''
    } else if (typeof VLY === 'string') {
        if (VLY.includes('B') || VLY.includes('M') || VLY.includes('K')) {
            return 'fa fa-arrow-up'
        } else {
            VLY = parseFloat(VLY)
            if (typeof VLY === 'number') {
                if (VLY > 0) {
                    return 'fa fa-arrow-up'
                } else {
                    return 'fa fa-arrow-down'
                }
            }
        }
    } else {
        return ''
    }
}

const costCardsArrowIndicator = (variance) => {
    let VLY = variance.toString();
    if (VLY === '0') {
        return ''
    } else if (typeof VLY === 'string') {
        if (VLY.includes('B') || VLY.includes('M') || VLY.includes('K')) {
            return 'fa cost-up fa-arrow-up'
        } else {
            VLY = parseFloat(VLY)
            if (typeof VLY === 'number') {
                if (VLY > 0) {
                    return 'fa cost-up fa-arrow-up'
                } else {
                    return 'fa cost-down fa-arrow-down'
                }
            }
        }
    } else {
        return ''
    }
}

const isEmptyGraphData = (data) => {
    let isZero = 0;
    data.map((d, i) => d.value === 0 ? isZero = isZero + 1 : null)
    return isZero === data.length;
}

const convertNumberFormatWithDecimals = (number) => {
    var num = number || number === 0 ? number.toString() : ''
    var isPoint = num.includes('.') ? true : false
    var convertedNumber = '';
    if (num.includes("B")) {
        var newnum = num.substring(0, num.length - 1);
        var decimalCount = window.countDecimals(newnum)
        var zeros = isPoint ? decimalCount == 2 ? '0000000' : '00000000' : '000000000'
        convertedNumber = newnum + zeros
    } else if (num.includes("M")) {
        var newnum = num.substring(0, num.length - 1);
        var decimalCount = window.countDecimals(newnum)
        var zeros = isPoint ? decimalCount == 2 ? "0000" : "00000" : "000000"
        convertedNumber = newnum + zeros
    } else if (num.includes("K")) {
        var newnum = num.substring(0, num.length - 1);
        var decimalCount = window.countDecimals(newnum)
        var zeros = isPoint ? decimalCount == 2 ? "0" : "00" : "000"
        convertedNumber = newnum + zeros
    } else {
        convertedNumber = num
    }
    return parseInt(convertedNumber.replace('.', ''));
}

const formatDate = (d) => {
    let month = '' + (d.getMonth() + 1);
    let day = '' + d.getDate();
    let year = d.getFullYear();

    if (month.length < 2)
        month = '0' + month;
    if (day.length < 2)
        day = '0' + day;

    return [year, month, day].join('-');
}

const convertPOSAccess = (access) => {
    let pos_access = access.split('#')
    let base_access = 'Network'

    if (pos_access[1] !== undefined) {
        if (pos_access[1] !== '*') {
            base_access = removeQuotes(pos_access[1])
        }
    }
    if (pos_access[2] !== undefined) {
        if (pos_access[2] !== '*') {
            base_access = `${base_access} > ${removeQuotes(pos_access[2])}`
        }
    }
    if (pos_access[3] !== undefined) {
        if (pos_access[3] !== '*') {
            base_access = `${base_access} > ${removeQuotes(pos_access[3])}`
        }
    }
    return base_access;
}

const convertRouteAccess = (access) => {
    let route_access = access !== null ? access : {}
    let base_access = 'Network';

    if (Object.keys(route_access).length > 0) {
        if ((route_access).hasOwnProperty('selectedRouteGroup')) {
            base_access = `${route_access['selectedRouteGroup'].join(',')}`
        }
        if ((route_access).hasOwnProperty('selectedRouteRegion')) {
            base_access = `${base_access} > ${route_access['selectedRouteRegion'].join(',')}`
        }
        if ((route_access).hasOwnProperty('selectedRouteCountry')) {
            base_access = `${base_access} > ${route_access['selectedRouteCountry'].join(',')}`
        }
        if ((route_access).hasOwnProperty('selectedRoute')) {
            base_access = `${base_access} > ${route_access['selectedRoute'].join(',')}`
        }
    }
    return base_access;
}

const convertDateTime = (dateTime) => {
    const date = new Date(`${dateTime}`)
    return date.toUTCString()
}

const getPOSFiltersSearchURL = () => {
    let posFiltersSearchURL = ''
    let regionId = window.localStorage.getItem('RegionSelected')
    let countryId = window.localStorage.getItem('CountrySelected')
    let cityId = window.localStorage.getItem('CitySelected')
    let commonOD = window.localStorage.getItem('ODSelected')
    if (regionId !== null && regionId !== 'Null') {
        regionId = JSON.parse(regionId)
        posFiltersSearchURL = `?Region=${removeQuotes(regionId)}`
    }
    if (countryId !== null && countryId !== 'Null') {
        countryId = JSON.parse(countryId)
        posFiltersSearchURL = `?Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}`
    }
    if (cityId !== null && cityId !== 'Null') {
        cityId = JSON.parse(cityId)
        posFiltersSearchURL = `?Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}&POS=${removeQuotes(cityId)}`
    }
    if (commonOD !== null && commonOD !== 'Null' && commonOD !== '') {
        posFiltersSearchURL = `?Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}&POS=${removeQuotes(cityId)}&${encodeURIComponent('O&D')}=${removeQuotes(commonOD)}`
    }
    return posFiltersSearchURL
}

const getRouteFiltersSearchURL = () => {
    let routeGroup = window.localStorage.getItem('RouteGroupSelected')
    routeGroup = routeGroup !== null ? JSON.parse(routeGroup).join(',') : 'Network'
    let regionId = window.localStorage.getItem('RouteRegionSelected')
    let countryId = window.localStorage.getItem('RouteCountrySelected')
    let routeId = window.localStorage.getItem('RouteSelected')
    let leg = window.localStorage.getItem('LegSelected')
    let flight = window.localStorage.getItem('FlightSelected')
    let url = `?RouteGroup=${routeGroup}`;

    if (regionId !== null && regionId !== 'Null') {
        regionId = JSON.parse(regionId)
        url = `?RouteGroup=${routeGroup}&Region=${removeQuotes(regionId)}`
    }
    if (countryId !== null && countryId !== 'Null') {
        countryId = JSON.parse(countryId)
        url = `?RouteGroup=${routeGroup}&Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}`
    }
    if (routeId !== null && routeId !== 'Null') {
        routeId = JSON.parse(routeId)
        url = `?RouteGroup=${routeGroup}&Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}&Route=${removeQuotes(routeId)}`
    }
    if (leg !== null && leg !== 'Null' && leg !== '') {
        url = `?RouteGroup=${routeGroup}&Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}&Route=${removeQuotes(routeId)}&Leg=${removeQuotes(leg)}`
    }
    if (flight !== null && flight !== 'Null' && flight !== '') {
        url = `?RouteGroup=${routeGroup}&Region=${removeQuotes(regionId)}&Country=${removeQuotes(countryId)}&Route=${removeQuotes(routeId)}&Leg=${removeQuotes(leg)}&Flight=${removeQuotes(flight)}`
    }
    return url;
}

const DrillDownLevel = (regionId, countryId, cityId) => {
    let drilldown_level = 'network'
    if (regionId && regionId !== '*') {
        drilldown_level = 'region';
    }
    if (countryId && countryId !== '*') {
        drilldown_level = 'country';
    }
    if (cityId && cityId !== '*') {
        drilldown_level = 'pos';
    }
    return drilldown_level;
}

const dateArray = (endDate) => {
    return endDate ? endDate.split('-') : ''
}

const endDateWithFirst = (endDate) => {
    let endDateArray = endDate.split('-')
    endDateArray.length = 2
    let endDateWithFirst = [...endDateArray, ...["01"]]
    return endDateWithFirst.join('-')
}

const getDateFormat = (date) => {
    return `${window.shortMonthNumToName(date.getMonth() + 1)} ${date.getDate()}, ${date.getFullYear()}`
}

const getMonthWithDropdown = (monthData) => {
    const currentMonth = new Date(subDays(new Date(), 5)).getMonth() + 1
    const currentYear = new Date(subDays(new Date(), 5)).getFullYear()
    let monthWithDropdown = monthData.filter((data) => {
        const monthNumber = window.monthNameToNum(data.MonthName)
        const year = parseInt(data.Year)
        if (year < currentYear) {
            if (data.Month.includes('▼')) {
                data.Month = `► ${(data.Month).substring(2, data.Month.length)}`
            } else if (!data.Month.includes('►')) {
                data.Month = `► ${data.Month}`
            }
        } else if (year === currentYear) {
            if (monthNumber <= currentMonth) {
                if (data.Month.includes('▼')) {
                    data.Month = `► ${(data.Month).substring(2, data.Month.length)}`
                } else if (!data.Month.includes('►')) {
                    data.Month = `► ${data.Month}`
                }
            }
        }
        data.highlightMonth = false
        return data;
    })
    return monthWithDropdown;
}

const getAlertMonthWithDropdown = (monthData) => {
    monthData.forEach((data) => {
        if (data.Month.includes('▼')) {
            data.Month = `► ${(data.Month).substring(2, data.Month.length)}`
        } else if (!data.Month.includes('►')) {
            data.Month = `► ${data.Month}`
        }
        data.highlightMonth = false
    })

    return monthData;
}

const getActionWithDropdown = (drilldownData) => {
    let actionWithDropdown = drilldownData.filter((data) => {
        data.firstColumnName = String(data.firstColumnName)
        if (data.firstColumnName.includes('▼')) {
            data.firstColumnName = `► ${(data.firstColumnName).substring(2, data.firstColumnName.length)}`
        } else if (!data.firstColumnName.includes('►')) {
            data.firstColumnName = `► ${data.firstColumnName}`
        }
        data.highlightMonth = false
        return data;
    })
    return actionWithDropdown;
}


const getStartEndDateOfMonth = (month, year) => {
    return {
        'startDate': formatDate(new Date(year, month - 1, 1)),
        'endDate': formatDate(new Date(year, month - 1 + 1, 0)),
    }
}

const isOdd = (num) => {
    if (num % 2 === 0)
        return false;
    return true;
}

const addZeroInMonth = (date) => {
    let dateArray = date ? date.split('-') : null;
    let fullDate = date
    if (date) {
        if (dateArray[1].toString().length !== 2) {
            fullDate = `${dateArray[0]}-0${dateArray[1]}-${dateArray[2]}`
        }
    }
    return fullDate;
}

const debounce = (func, timeout = 300) => {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => { func.apply(this, args); }, timeout);
    };
}

export default {
    loggedinUser,
    sendEvent,
    validateEmail,
    validatePassword,
    validatePhone,
    validateNum,
    validateCharNum,
    validateCity,
    validateUsername,
    removeQuotes,
    addQuotesforMultiSelect,
    getParameterByName,
    normalizeLineChartData,
    normalizeData,
    rectifiedDeltaData,
    cardsArrowIndicator,
    costCardsArrowIndicator,
    isEmptyGraphData,
    convertNumberFormatWithDecimals,
    formatDate,
    convertPOSAccess,
    convertRouteAccess,
    convertDateTime,
    getPOSFiltersSearchURL,
    getRouteFiltersSearchURL,
    DrillDownLevel,
    dateArray,
    endDateWithFirst,
    getDateFormat,
    isOdd,
    getMonthWithDropdown,
    getAlertMonthWithDropdown,
    getActionWithDropdown,
    getStartEndDateOfMonth, 
    addZeroInMonth,
    debounce
};