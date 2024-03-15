import React, { Component } from 'react';
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Loader from '../../Component/Loader';
import Access from "../../Constants/accessValidation";
import Validator from "../../Constants/validator";
import api from '../../API/api'
import './Alerts.scss';
import Slider from '@material-ui/core/Slider';
import Spinners from "../../spinneranimation";
import _ from 'lodash';
import Swal from 'sweetalert2';
import cookieStorage from '../../Constants/cookie-storage';
import { string } from '../../Constants/string';


const apiServices = new APIServices();


class AlertSetting extends Component {
    constructor(props) {
        super(props);
        this.state = {
            forecast: {},
            booking: {},
            availability: {},
            market_share: {},
            forecastRange: [],
            bookingRange: [],
            availabilityRange: [],
            market_shareRange: [],
            pos: {},
            posDays: null,
            disable: true,
            infare: [],
            carrierRate: null,
            disableBtn: false,
            posErrorMsg: '',
            carrierErrorMsg: ''

        };

    }

    componentWillMount() {
        if (Access.accessValidation('Alert', 'alertSetting')) {
            this.getAlertSettings()
        } else {
            this.props.history.push('/404')
        }
    }

    getAlertSettings = () => {
        api.get(`getAlertsSettings`)
            .then((response) => {
                if (response) {
                    if (response.data.response.length > 0) {
                        const data = response.data.response;
                        let forecast = data.filter((d) => d.label === 'forecast')
                        let booking = data.filter((d) => d.label === 'booking')
                        let availability = data.filter((d) => d.label === 'availability')
                        let market_share = data.filter((d) => d.label === 'market_share')
                        let pos = data.filter((d) => d.label === 'pos')
                        let infare = data.filter((d) => d.label === 'Infare')
                        this.setState({
                            forecast: forecast[0],
                            booking: booking[0],
                            availability: availability[0],
                            market_share: market_share[0],
                            forecastRange: forecast[0].range,
                            bookingRange: booking[0].range,
                            availabilityRange: availability[0].range,
                            market_shareRange: market_share[0].range,
                            pos: pos[0],
                            posDays: pos[0].range[0],
                            infare: infare[0],
                            carrierRate: infare[0].range[0]
                        });
                    }
                }
            })
            .catch((err) => {
                console.log('alert setting error', err)
            })
    }

    handleChange(e) {
        this.setState({ [e.target.name]: e.target.value })
        this.validate(e)
        setTimeout(() => {
            this.validateForm()
        }, 500);
    }

    handleRangeChange = (name) => (event, newValue) => {
        this.setState({ [name]: newValue })
    }

    validate(e) {
        const { posDays, carrierRate } = this.state;
        const name = e.target.name
        const value = e.target.value
        console.log('rahul', posDays, carrierRate)
        if (name === 'posDays') {
            if (!value) {
                this.setState({ posErrorMsg: 'Pos Days cannot be empty' })
            }
            else if (!Validator.validateNum(value)) {
                this.setState({ posErrorMsg: 'Enter Numbers Only' })
            } else {
                this.setState({ posErrorMsg: '' })
            }
        }
        if (name === 'carrierRate') {
            if (!value) {
                this.setState({ carrierErrorMsg: 'Carrier rate cannot be empty' })
            }
            else if (!Validator.validateNum(value)) {
                this.setState({ carrierErrorMsg: 'Enter Numbers Only' })
            } else {
                this.setState({ carrierErrorMsg: '' })
            }
        }
    }

    validateForm() {
        const { posErrorMsg, carrierErrorMsg } = this.state;
        if (posErrorMsg || carrierErrorMsg) {
            this.setState({ disableBtn: true })
        } else {
            this.setState({ disableBtn: false })
        }
    }

    updateAlertSettings = () => {
        const { forecast, booking, availability, market_share, pos, infare, forecastRange, bookingRange, availabilityRange, market_shareRange, posDays, carrierRate } = this.state;
        const userData = JSON.parse(cookieStorage.getCookie('userDetails'))
        var postData = {
            "user_id": userData.id,
            "settings": [{
                "id": forecast.id,
                "threshold_start_limit": forecastRange[0],
                "threshold_end_limit": forecastRange[1]
            },
            {
                "id": booking.id,
                "threshold_start_limit": bookingRange[0],
                "threshold_end_limit": bookingRange[1]
            },
            {
                "id": availability.id,
                "threshold_start_limit": availabilityRange[0],
                "threshold_end_limit": availabilityRange[1]
            },
            {
                "id": market_share.id,
                "threshold_start_limit": market_shareRange[0],
                "threshold_end_limit": market_shareRange[1]
            },
            {
                "id": pos.id,
                "threshold_start_limit": posDays,
                "threshold_end_limit": null
            },
            {
                "id": infare.id,
                "threshold_start_limit": carrierRate,
                "threshold_end_limit": null
            }]
        };
        api.post(`updateAlertsSettings`, postData)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `Alert settings updated successfully`,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    // this.getAlertSettings()
                })

            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to update Alert settings`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    avoidE(e) {
        console.log('rahul', e.keyCode !== 69)
        return e.keyCode !== 69
    }

    render() {
        const { forecastRange, bookingRange, availabilityRange, market_shareRange, posDays, carrierRate, disableBtn, posErrorMsg, carrierErrorMsg } = this.state;
        return (
            <div>
                <Loader />
                <TopMenuBar {...this.props} />
                <div className='alerts'>
                    <div className='add'>
                        <h2>Alert Setting</h2>
                    </div>
                    <div className='alerts-setting-form'>

                        <div class="form-group">
                            <label >{`Forecast :`}</label>
                            <div className='slider'>
                                <span>0%</span>
                                <span className='end'>100%</span>
                                <Slider
                                    value={forecastRange}
                                    onChange={this.handleRangeChange('forecastRange')}
                                    valueLabelDisplay="auto"
                                    aria-labelledby="range-slider"
                                />
                            </div>
                        </div>

                        <div class="form-group">
                            <label >{`Booking :`}</label>
                            <div className='slider'>
                                <span>0%</span>
                                <span className='end'>100%</span>
                                <Slider
                                    value={bookingRange}
                                    onChange={this.handleRangeChange('bookingRange')}
                                    valueLabelDisplay="auto"
                                    aria-labelledby="range-slider"
                                />
                            </div>
                        </div>

                        <div class="form-group">
                            <label >{`Availability :`}</label>
                            <div className='slider'>
                                <span>0%</span>
                                <span className='end'>100%</span>
                                <Slider
                                    value={availabilityRange}
                                    onChange={this.handleRangeChange('availabilityRange')}
                                    valueLabelDisplay="auto"
                                    aria-labelledby="range-slider"
                                />
                            </div>
                        </div>

                        <div class="form-group">
                            <label >{`Market Share :`}</label>
                            <div className='slider'>
                                <span>0%</span>
                                <span className='end'>100%</span>
                                <Slider
                                    value={market_shareRange}
                                    onChange={this.handleRangeChange('market_shareRange')}
                                    valueLabelDisplay="auto"
                                    aria-labelledby="range-slider"
                                />
                            </div>
                        </div>

                        <div class="form-group">
                            <label style={{ marginBottom: '5px' }}>POS :</label>
                            <input type="text" class="form-control" placeholder='Set POS Days' name="posDays" value={posDays} onChange={(e) => this.handleChange(e)} />
                            <p>{posErrorMsg}</p>
                        </div>

                        <div class="form-group">
                            <label style={{ marginBottom: '5px' }}>Infare :</label>
                            <input type="text" class="form-control" placeholder='Set Carrier Rate' name='carrierRate' value={carrierRate} onChange={(e) => this.handleChange(e)} />
                            <p>{carrierErrorMsg}</p>
                        </div>

                        <div className='action-btn'>
                            <button type="button" className="btn btn-danger" onClick={() => this.updateAlertSettings()} disabled={disableBtn}>Update</button>
                        </div>

                    </div>
                </div>
            </div>
        );
    }
}

export default AlertSetting;