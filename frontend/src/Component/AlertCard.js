import React, { Component } from 'react';
import api from '../API/api'
import Swal from 'sweetalert2';
import cookieStorage from '../Constants/cookie-storage';
import RejectAccess from '../Constants/accessValidation';
import './component.scss'

class AlertCard extends Component {
    constructor(props) {
        super(props);
        this.state = {
            Aknowledge: [],
            Reject: [],
            acknowledgedAlerts: [],
            alertData: [],
        }
    }

    componentWillMount() {
        this.userData = JSON.parse(cookieStorage.getCookie('userDetails'))
    }

    acknowledge = (alertID) => {
        const postData = {
            "alert_id": alertID,
            "user_id": this.userData.id,
            "acknowledged": "1",
            "rejected": "0"
        }
        api.post(`updatealertsdetails`, postData)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `Alert aknowledged successfully `,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    let Aknowledge = this.state.Aknowledge;
                    Aknowledge.push(alertID)
                    this.setState({ Aknowledge: Aknowledge })
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to aknowledge alert `,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    reject = (alertID) => {
        const postData = {
            "alert_id": alertID,
            "user_id": this.userData.id,
            "acknowledged": "0",
            "rejected": "1"
        }
        api.post(`updatealertsdetails`, postData)
            .then((res) => {
                Swal.fire({
                    title: 'Success!',
                    text: `Alert rejected successfully `,
                    icon: 'success',
                    confirmButtonText: 'Ok'
                }).then(() => {
                    let Reject = this.state.Reject;
                    Reject.push(alertID)
                    this.setState({ Reject: Reject })
                })
            })
            .catch((err) => {
                Swal.fire({
                    title: 'Error!',
                    text: `Unable to reject alert`,
                    icon: 'error',
                    confirmButtonText: 'Ok'
                })
            })
    }

    render() {
        const { alertData, Aknowledge, Reject } = this.state;
        const { modal, data } = this.props;
        return (
            data.map((data, i) =>
                <div className='alert-card-main'>
                    <div className={`card ${(data.aul && data.aul.length > 0) || (Aknowledge.includes(data.id) || Reject.includes(data.id)) ? 'fade-card' : ''}`}>
                        <div className='heading'>
                            {`OD : ${data.CommonOD}`}
                        </div>
                        <h5 className='action'>{data.Action}</h5>
                        <div className='details'>
                            <ul>
                                {/* <li>{`Booking VS Last Year is`}<span className={`${data.BStatus === 'Above' ? 'green' : 'red'}`}>{` ${data.BStatus}`}</span></li>
                                <li>{`Forecast VS Target is `}<span className={`${data.FStatus === 'Above' ? 'green' : 'red'}`}>{`${data.FStatus}`}</span></li>
                                <li>{`Availability in Lower Class is `}<span className={`${data.AStatus === 'Above' ? 'green' : 'red'}`}>{`${data.AStatus}`}</span></li>
                                <li>{`Market Share is `}<span className={`${data.MStatus === 'High' ? 'green' : 'red'}`}>{`${data.MStatus}`}</span></li> */}
                                <li>{`Infare rates are `}<span className={`${data.IStatus === 'High' ? 'green' : 'red'}`}>{`${data.IStatus}`}</span></li>
                            </ul>
                        </div>
                        {(data.aul && data.aul.length > 0) || (Aknowledge.includes(data.id) || Reject.includes(data.id)) ?
                            <h5>
                                {` ${data.aul && data.aul.length > 0 ?
                                    data.aul[0].acknowledged === '1' ? 'Aknowledged' : data.aul[0].rejeceted === '1' ? 'Rejected' : ''
                                    :
                                    Aknowledge.includes(data.id) ? 'Aknowledged' : Reject.includes(data.id) ? 'Rejected' : ``}`}
                            </h5>
                            :
                            <div className='card-btn'>
                                <button type="submit" onClick={() => this.acknowledge(data.id)} className="btns">
                                    Acknowledge
                            </button>
                                {RejectAccess.isRejectAlert() ?
                                    <button type="submit" onClick={() => this.reject(data.id)} className="btns">
                                        Reject
                                 </button> : ''}
                            </div>}
                    </div>
                </div>
            )
        );
    }

}

export default AlertCard;