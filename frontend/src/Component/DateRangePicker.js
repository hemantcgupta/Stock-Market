import React, { Component } from 'react';
import { DateRangePicker } from 'react-date-range';
import 'react-date-range/dist/styles.css'; // main style file
import 'react-date-range/dist/theme/default.css'; // theme css file
import { addDays, subMonths, addMonths } from 'date-fns';

class DatePicker extends Component {
    constructor(props) {
        super(props)

    }

    render() {
        const { showDatePicker, dateSelected, onClose, handleDatePicker, datePickerValue, center } = this.props;
        return (

            showDatePicker ? <div className={`dateRangePicker ${center ? 'date-range-picker-center' : ''}`}>
                <div className='btn-container' >
                    <button className='select-date-btn' style={{ width: '60px' }} onClick={dateSelected}>Ok</button>
                    <button className='select-date-btn' style={{ marginLeft: '95px' }} onClick={onClose}>Cancel</button>
                </div>
                <DateRangePicker
                    onChange={handleDatePicker}
                    showSelectionPreview={true}
                    moveRangeOnFirstSelection={false}
                    months={2}
                    ranges={datePickerValue}
                    minDate={new Date(new Date().getFullYear() - 1, 0, 1)}
                    maxDate={addMonths(new Date(new Date().getFullYear(), new Date().getMonth(), 1), 12)}
                    dateDisplayFormat={`MMM d, yyyy`}
                    direction="horizontal"
                />
            </div> : <div />

        )
    }

}

export default DatePicker;
