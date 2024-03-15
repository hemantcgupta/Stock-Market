import React, { Component } from 'react';
import { Redirect } from "react-router-dom";
import APIServices from '../../API/apiservices';
import TopMenuBar from '../../Component/TopMenuBar';
import Pagination from '../../Component/pagination';
import DatePicker from '../../Component/DatePicker';
import Access from "../../Constants/accessValidation";
import { addDays, subDays } from 'date-fns';
import AlertCard from '../../Component/AlertCard';
import Modal from 'react-bootstrap-modal';
import Spinners from "../../spinneranimation";
import Swal from 'sweetalert2';
import api from '../../API/api'
import cookieStorage from '../../Constants/cookie-storage';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/dist/styles/ag-grid.css';
import 'ag-grid-community/dist/styles/ag-theme-balham.css';
import '../../ag-grid-table.scss';
import './Alerts.scss';

const apiServices = new APIServices();

class AuditLogs extends Component {
    constructor(props) {
        super(props);
        this.gridApi = null;
        this.state = {
            alert: '',
            date: 'Select Date Range',
            showDatePicker: false,
            datePickerValue: [{
                startDate: subDays(new Date(), 10),
                endDate: new Date(),
                key: 'selection'
            }],
            alertData: [],
            alertColumns: [
                { headerName: 'Region', field: 'Region', alignLeft: true },
                { headerName: 'Country', field: 'Country', alignLeft: true },
                { headerName: 'POS', field: 'City', alignLeft: true },
                { headerName: 'O&D', field: 'O&D', alignLeft: true },
                // { headerName: 'Availability Status', headerTooltip: 'Availability Status', field: 'AStatus', tooltipField: 'AStatus', cellRenderer: (params) => this.arrowIndicator(params) },
                // { headerName: 'Booking Status', headerTooltip: 'Booking Status', field: 'BStatus', tooltipField: 'BStatus', cellRenderer: (params) => this.arrowIndicator(params) },
                // { headerName: 'Forecast Status', headerTooltip: 'Forecast Status', field: 'FStatus', tooltipField: 'FStatus', cellRenderer: (params) => this.arrowIndicator(params) },
                { headerName: 'Infare Rates', headerTooltip: 'Infare Rates', field: 'IStatus', tooltipField: 'device_type', cellRenderer: (params) => this.arrowIndicator(params) },
                // { headerName: 'Market Share', headerTooltip: 'Market Share', field: 'MStatus', tooltipField: 'MStatus', cellRenderer: (params) => this.arrowIndicator(params) },
                // { headerName: 'Criticality', field: 'Criticality', tooltipField: 'Criticality' },
                // { headerName: 'Strategy', field: 'Stratagy', tooltipField: 'Stratagy' },
                {
                    headerName: 'Actions', field: 'Actions', tooltipField: 'Actions',
                    cellStyle: (params) => {
                        return {
                            'text-decoration': 'underline',
                            'cursor': 'pointer'
                        };
                    }
                }
            ],
            AStatus: '',
            FStatus: '',
            BStatus: '',
            MStatus: '',
            IStatus: '',
            action: '',
            startDate: '',
            endDate: '',
            action: '',
            device: '',
            currentPage: '',
            totalPages: '',
            totalRecords: '',
            paginationStart: 1,
            paginationEnd: '',
            paginationSize: '',
            count: 1,
            loading: false,
            selectedAlertId: null,
            modalVisible: false,
            isAcknowledged: false,
            isRejected: false
        };

    }

    componentDidMount() {
        if (Access.accessValidation('Alert', 'alerts')) {
            const defaultDate = this.state.datePickerValue[0];
            const startDate = defaultDate.startDate.toDateString();
            const endDate = defaultDate.endDate.toDateString();
            this.setState({
                date: `${startDate} - ${endDate}`,
                startDate: defaultDate.startDate.toJSON(),
                endDate: defaultDate.endDate.toJSON()
            },
                // () => this.gotoFirstPage()
            )
            this.getCardsData()
        } else {
            this.props.history.push('/404')
        }
    }

    arrowIndicator = (params) => {
        var element = document.createElement("span");
        var icon = document.createElement("i");

        // visually indicate if this months value is higher or lower than last months value
        let value = params.value;
        if (value === 'High' || value === 'Above') {
            icon.className = 'fa fa-arrow-up'
        } else {
            icon.className = 'fa fa-arrow-down'
        }

        element.appendChild(document.createTextNode(params.value));
        element.appendChild(icon);
        return element;
    }

    handleChange = (e) => {
        this.setState({ [e.target.id]: e.target.value });
    }

    callCriticality = (e) => {
        this.setState({ selectedCriticality: e.target.value })
    }

    handleDatePicker = (item) => {
        this.setState({ datePickerValue: [item.selection] })
    }

    dateSelected() {
        const datePickerValue = this.state.datePickerValue;
        const startDate = datePickerValue[0].startDate.toDateString();
        const endDate = datePickerValue[0].endDate.toDateString();
        this.setState({
            showDatePicker: false,
            date: `${startDate} - ${endDate}`,
            startDate: datePickerValue[0].startDate.toJSON(),
            endDate: datePickerValue[0].endDate.toJSON()
        })
    }

    getCardsData = () => {
        this.setState({ loading: true, alertData: [] })
        const { AStatus, BStatus, FStatus, IStatus, MStatus, action } = this.state;
        const params = {};

        if (AStatus) {
            params['AStatus'] = AStatus
        }
        if (BStatus) {
            params['BStatus'] = BStatus
        }
        if (FStatus) {
            params['FStatus'] = FStatus
        }
        if (IStatus) {
            params['IStatus'] = IStatus
        }
        if (MStatus) {
            params['MStatus'] = MStatus
        }
        if (action) {
            params['actions'] = action
        }
        const serialize = (params) => {
            var str = [];
            for (var p in params)
                if (params.hasOwnProperty(p)) {
                    str.push(encodeURIComponent(p) + "=" + encodeURIComponent(params[p]));
                }
            return str.join("&");
        }

        api.get(`getallalerts?${serialize(params)}`)
            .then((response) => {
                this.setState({ loading: false })
                if (response) {
                    if (response.data.response.length > 0) {
                        let data = response.data.response;
                        let alertData = [];
                        data.forEach(function (key) {
                            alertData.push({
                                'id': key.id,
                                "Region": key.Region,
                                "Country": key.CountryCode,
                                "City": key.POS,
                                "O&D": key.CommonOD,
                                "AStatus": key.AStatus,
                                'BStatus': key.BStatus,
                                "Criticality": key.Criticality,
                                "FStatus": key.FStatus,
                                "IStatus": key.IStatus,
                                'MStatus': key.MStatus,
                                'Stratagy': key.Stratagy,
                                'Actions': 'Show More',
                                'aul': key.aul
                            });
                        });
                        this.setState({
                            alertData: alertData
                        });
                    }
                }
            })
            .catch((err) => {
                this.setState({ loading: false })
                Swal.fire({
                    title: "Error!",
                    text: "something went wrong, please try again!",
                    icon: "error",
                    confirmButtonText: 'Ok'
                })
            })
    }

    onCellClicked = (params) => {
        let column = params.colDef.field;
        let id = params.data.id;
        if (column === 'Actions') {
            this.setState({ modalVisible: true, selectedAlertId: parseInt(id), isAcknowledged: false, isRejected: false },
                () => {
                    let selectedAlert = this.state.alertData.filter((d) => d.id === this.state.selectedAlertId)
                    selectedAlert.map((d) => d.aul ? d.aul.map((a) => a.acknowledged === '1' ? this.setState({ isAcknowledged: true }) : null) : null)
                    selectedAlert.map((d) => d.aul ? d.aul.map((a) => a.rejeceted === '1' ? this.setState({ isRejected: true }) : null) : null)
                })
        }
    }

    closeModal() {
        this.setState({ modalVisible: false })
    }

    firstDataRendered = (params) => {
        params.api.sizeColumnsToFit();
    }

    applyColumnLockDown(columnDefs) {
        return columnDefs.map((c) => {
            if (c.children) {
                c.children = this.applyColumnLockDown(c.children);
            }
            const cellClass = {}
            const alreadyCellBasis = c.cellClassRules;
            if (c.alignLeft) {
                cellClass.cellClassRules = {
                    'align-left': '1 == 1',
                    ...alreadyCellBasis,
                };
            } else {
                cellClass.cellClassRules = {
                    'align-right': '1 == 1',
                    ...alreadyCellBasis,
                };
            }
            return {
                ...c, ...cellClass, ...{
                    lockPosition: true,
                    cellClass: 'lock-pinned',

                }
            }
        }
        )
    }

    onGridReady = (params) => {
        if (!params.api) {
            return null;
        }
        this.gridApi = params.api;
    }

    render() {
        const { alertColumns, alertData, selectedAlertId, isAcknowledged, isRejected } = this.state;
        setInterval(() => {
            if (this.gridApi) {
                if (this.gridApi.gridPanel.eBodyViewport.clientWidth) {
                    this.gridApi.sizeColumnsToFit();
                }
            }
        }, 1000)

        return (
            <div>
                <TopMenuBar {...this.props} />
                <div className='alerts'>
                    <div className='add'>
                        <h2>Alerts</h2>
                    </div>
                    <div className='alerts-form'>
                        {/* <div class="form-group">
                            <label for="alert">Alert Type :</label>
                            <input type="text" class="form-control" id="alert" value={this.state.alert} onChange={(e) => this.handleChange(e)} />
                        </div>
                        <div class="form-group">
                            <label for="datePicker">Date Picker :</label>
                            <button className="form-control cabinselect dashboard-dropdown date-picker-btn" onClick={() => this.setState({ showDatePicker: !this.state.showDatePicker })}>{this.state.date}</button>
                            {this.state.showDatePicker ? <div className='triangle'><div className="triangle-up"></div></div> : <div />}
                        </div> */}
                        {/* <div class="form-group">
                            <label for="events">Availability Status :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="AStatus">
                                <option selected={true} value="">All</option>
                                <option value="Above">Above</option>
                                <option value="Below">Below</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="events">Booking Status :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="BStatus">
                                <option selected={true} value="">All</option>
                                <option value="Above">Above</option>
                                <option value="Below">Below</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="events">Forecast Status :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="FStatus">
                                <option selected={true} value="">All</option>
                                <option value="Above">Above</option>
                                <option value="Below">Below</option>
                            </select>
                        </div> */}
                        <div class="form-group">
                            <label for="events">Infare Rates :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="IStatus">
                                <option selected={true} value="">All</option>
                                <option value="High">High</option>
                                <option value="Low">Low</option>
                            </select>
                        </div>
                        {/* <div class="form-group">
                            <label for="events">Market Share :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="MStatus">
                                <option selected={true} value="">All</option>
                                <option value="High">High</option>
                                <option value="Low">Low</option>
                            </select>
                        </div> */}
                        <div class="form-group">
                            <label for="events">Actions :</label>
                            <select className="form-control cabinselect events-dropdown"
                                onChange={(e) => this.handleChange(e)} id="action">
                                <option selected={true} value="">All</option>
                                <option value="Acknowledged">Acknowledged</option>
                                <option value="Rejected">Rejected</option>
                            </select>
                        </div>
                        <button type="button" className="btn btn-danger" onClick={() => this.getCardsData()}>Search</button>
                    </div>
                    {/* <div className='alerts-card'>
                        {this.state.loading ?
                            <Spinners /> :
                            <div>
                                <div className='card-main'>
                                    {this.state.alertData.length > 0 ? <AlertCard data={this.state.alertData} /> : <h3 style={{ width: '100%', textAlign: 'center' }}>No Alerts to show</h3>}
                                </div>
                            </div>
                        }
                    </div> */}
                    <div className='audit-logs-table'>
                        {this.state.loading ?
                            <Spinners /> :
                            <div
                                id='myGrid' className={`ag-theme-balham-dark ag-grid-table`}
                                style={{ height: 'calc(100vh - 350px)' }}
                            >
                                <AgGridReact
                                    onGridReady={this.onGridReady}
                                    onFirstDataRendered={(params) => this.firstDataRendered(params)}
                                    columnDefs={this.applyColumnLockDown(alertColumns)}
                                    rowData={alertData}
                                    onCellClicked={(params) => this.onCellClicked(params)}
                                    rowSelection={`single`}
                                    enableBrowserTooltips={true}
                                >
                                </AgGridReact>
                            </div>}
                    </div>
                </div>
                {/* <DatePicker
                    showDatePicker={this.state.showDatePicker}
                    dateSelected={() => this.dateSelected()}
                    onClose={() => this.setState({ showDatePicker: false })}
                    handleDatePicker={item => this.handleDatePicker(item)}
                    datePickerValue={this.state.datePickerValue}
                    center={true}
                /> */}
                <Modal
                    show={this.state.modalVisible}
                    onHide={() => this.closeModal()}
                    aria-labelledby="ModalHeader"
                >
                    <Modal.Header closeButton>
                        <Modal.Title id='ModalHeader'>{`Actions taken by users`}</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        {alertData.map((data, i) =>
                            data.id === selectedAlertId ?
                                data.aul && data.aul.length > 0 ?
                                    <div className='users-initials'>

                                        {isAcknowledged ? <h5>Acknowledged :</h5> : ''}
                                        <div className='users-initials-main'>
                                            {data.aul.map((ele) =>
                                                ele.acknowledged === '1' ?
                                                    <div title={`${ele.username}`} className='users-circle'>
                                                        <span className='initial'>{`${ele.username.charAt(0)}`}</span>
                                                    </div> : '')}
                                        </div>


                                        {isRejected ? <h5>Rejected :</h5> : ''}
                                        <div className='users-initials-main'>
                                            {data.aul.map((ele) =>
                                                ele.rejeceted === '1' ?
                                                    <div title={`${ele.username}`} className='users-circle'>
                                                        <span className='initial'>{`${ele.username.charAt(0)}`}</span>
                                                    </div> : '')}
                                        </div>
                                    </div> : <h5 style={{ textAlign: 'center' }}>No action taken yet</h5> : '')}
                    </Modal.Body>
                </Modal>
            </div>

        );
    }
}

export default AuditLogs;