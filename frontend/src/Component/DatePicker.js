
import React, { Component } from 'react';
import { DateRangePicker } from 'react-date-range';
import 'react-date-range/dist/styles.css'; // main style file
import 'react-date-range/dist/theme/default.css'; // theme css file
import { addDays } from 'date-fns';

import PropTypes from 'prop-types'
import Picker from 'react-month-picker'
import 'react-month-picker/css/month-picker.css';


class MonthBox extends Component {
    static propTypes = {
        value: PropTypes.string,
        onClick: PropTypes.func,
    }

    constructor(props, context) {
        super(props, context)

        this.state = {
            value: this.props.value || 'N/A',
        }
    }

    static getDerivedStateFromProps(props, state) {
        return {
            value: props.value || 'N/A',
        }
    }

    render() {

        return (
            <div className="box" onClick={this._handleClick}>
                <label>{this.state.value}</label>
            </div>
        )
    }

    _handleClick = (e) => {
        this.props.onClick && this.props.onClick(e)
    }
}

class DatePicker extends Component {
    constructor(props) {
        super(props);
        this.state = {
            currentMonth: new Date().getMonth() + 1,
            currentYear: new Date().getFullYear(),
        }
        this.pickRange = React.createRef()
    }


    _handleClickRangeBox = (e) => {
        this.pickRange.current.show()
    }

    handleRangeChange = (value, text) => {

    }

    render() {
        const pickerLang = {
            months: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            from: 'From', to: 'To',
        }
        const { rangeValue, handleRangeDissmis, dateRange, monthRange } = this.props

        const { currentMonth, currentYear } = this.state;

        const makeText = m => {
            if (m && m.year && m.month) return (pickerLang.months[m.month - 1] + '. ' + m.year)
            return '?'
        }

        return (
            <div className="edit">
                <Picker
                    ref={this.pickRange}
                    years={{
                        min: {
                            year: monthRange === 'nextYear' ? currentYear + 1 : currentYear - 1,
                            month: 1
                        },
                        max: {
                            year: monthRange === 'nextYear' ? currentYear + 1 : currentMonth === 1 ? currentYear : currentYear + 1,
                            month: monthRange === 'nextYear' ? 12 : currentMonth === 1 ? 12 : currentMonth - 1
                        }
                    }}
                    value={rangeValue}
                    lang={pickerLang}
                    theme="light"
                    onChange={this.handleRangeChange}
                    onDismiss={handleRangeDissmis}
                >
                    <MonthBox value={makeText(rangeValue.from) + ' ~ ' + makeText(rangeValue.to)} onClick={this._handleClickRangeBox} />
                </Picker>
            </div>
        )
    }
}

export default DatePicker;